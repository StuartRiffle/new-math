#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <array>
#include <memory>
#include <atomic> // For std::atomic

// This program performs a breadth first traversal of the Collatz state space.
//
// Combining small primes into one modulus allows the state to fit in a 64-bit word:
//
//  2 × 3 × 5 × 7 × 11 × 13 = 30,030
//
// The total state space is the product of the odd primes up to 47
//
//  × 17 =                   510,510
//  × 19 =                 9,699,690
//  × 23 =               223,092,870
//  × 29 =             6,469,693,230
//  × 31 =           200,560,490,130
//  x 37 =         7,419,488,134,810
//  x 41 =       304,199,012,527,210
//  × 43 =    13,080,570,038,669,930
//  × 47 =   614,786,056,966,487,710
//
// This comes out to about 614 quadrillion residue vectors. Tracking visited states
// requires one bit for each of them, so we need almost 78 TB of memory (uncompressed)
// to perform the full breadth first search.
//
// The full state space must be sharded to fit in working memory. Each slice of the
// bitset requires gigabytes of storage, depending on the granularity of sharding:
//
//        Shards   Bitset size
//  78 TB /  256 = 304 GB
//        /  512 = 152 GB
//        / 1024 =  76 GB
//        / 2048 =  38 GB
//        / 4096 =  19 GB
//        / 8192 =  10 GB
//
// Output buffers at runtime require about the same amount of memory.
//
// Every Collatz step mutates the state, and each shard represents an equal portion
// of the state space. Each step "scatters" all its active orbits to a different shard.
// The space is partitioned to balance thit output between all other shards on average.
// Incoming batches for a shard are stored in files, providing a simple work queue.
// The frontier of the search is pushed between shards in this way. The shards can
// be processed serially or in parallel, as all persistent state is stored in files.
//
// This process is heavily I/O bound. The computation itself is trivial, but the bitset
// is huge, and "message passing" of states involves a lot of bandwidth. This is a
// gather/scatter operation that must be performed thousands of times. 

#include <cuda_runtime.h>

// Helper to check for CUDA errors
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

constexpr size_t SMALL_PRIMES_PRODUCT = 3 * 5 * 7 * 11 * 13;
__constant__ uint32_t D_MODULI[]      = { SMALL_PRIMES_PRODUCT, 17, 19, 23, 29, 31, 37, 41, 43, 47 };
__constant__ uint32_t D_FIELD_BITS[]  = { 15, 5, 5, 5, 5, 5, 6, 6, 6, 6 };
__constant__ uint64_t D_FIELD_MASK[]  = { 0x7FFF, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x3F, 0x3F, 0x3F, 0x3F };

constexpr int NUM_FIELDS = 10;
using PackedState = uint64_t;

constexpr size_t SHARD_BITS = 8;
constexpr size_t NUM_SHARDS = (1ULL << SHARD_BITS);
constexpr size_t SHARD_SIZE = (1ULL << 20);
constexpr size_t BITSET_BYTES = SHARD_SIZE / 8;

constexpr int THREADS_PER_BLOCK = 256;


// -------------------- DEVICE HELPER FUNCTIONS --------------------

__device__ std::array<uint16_t, NUM_FIELDS> unpack_residues(PackedState state) {
    std::array<uint16_t, NUM_FIELDS> residues{};
    int shift = 0;
    for (int i = 0; i < NUM_FIELDS; ++i) {
        residues[i] = (state >> shift) & D_FIELD_MASK[i];
        shift += D_FIELD_BITS[i];
    }
    return residues;
}

__device__ PackedState pack_residues(const std::array<uint16_t, NUM_FIELDS>& residues) {
    uint64_t state = 0;
    int shift = 0;
    for (int i = 0; i < NUM_FIELDS; ++i) {
        state |= (uint64_t(residues[i]) & D_FIELD_MASK[i]) << shift;
        shift += D_FIELD_BITS[i];
    }
    return state;
}

__device__ std::array<uint16_t, NUM_FIELDS> collatz_residue_step(std::array<uint16_t, NUM_FIELDS> r) {
    if ((r[0] & 1) == 0) {
        for (int i = 0; i < NUM_FIELDS; ++i) {
            if (r[i] & 1) r[i] += D_MODULI[i];
            r[i] /= 2;
        }
    } else {
        for (int i = 0; i < NUM_FIELDS; ++i) {
            r[i] = (3 * r[i] + 1) % D_MODULI[i];
        }
        while ((r[0] & 1) == 0) {
            for (int i = 0; i < NUM_FIELDS; ++i) {
                if (r[i] & 1) r[i] += D_MODULI[i];
                r[i] /= 2;
            }
        }
    }
    return r;
}

__device__ size_t state_to_shard_id(PackedState s) {
    return s >> (64 - SHARD_BITS);
}

__device__ size_t state_to_idx_in_shard(PackedState s) {
    return s & (SHARD_SIZE - 1);
}

// -------------------- MAIN KERNEL --------------------

__global__ void process_states_kernel(
    const PackedState* work_queue,
    size_t num_states,
    uint8_t* visited_bitset,
    PackedState** outgoing_buckets,
    uint32_t* bucket_counters,
    size_t max_bucket_size,
    uint32_t* overflow_counter) // <-- NEW: To detect silent data loss
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) {
        return;
    }

    PackedState s = work_queue[tid];
    size_t idx = state_to_idx_in_shard(s);
    uint8_t mask = 1 << (idx % 8);
    uint8_t old_val = atomicOr(&visited_bitset[idx / 8], mask);

    if ((old_val & mask) != 0) {
        return;
    }

    std::array<uint16_t, NUM_FIELDS> residues = unpack_residues(s);
    residues = collatz_residue_step(residues);
    PackedState neighbor = pack_residues(residues);

    size_t target_shard = state_to_shard_id(neighbor);
    uint32_t append_idx = atomicAdd(&bucket_counters[target_shard], 1);

    if (append_idx < max_bucket_size) {
        outgoing_buckets[target_shard][append_idx] = neighbor;
    } else {
        // NEW: Atomically increment overflow counter if a bucket is full.
        atomicAdd(overflow_counter, 1);
    }
}

// -------------------- MAIN HOST FUNCTION --------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shard_dir>\n";
        return 1;
    }
    std::string shard_dir(argv[1]);

    std::vector<PackedState> h_work_queue;
    std::vector<std::filesystem::path> processed_files;
    for (const auto& entry : std::filesystem::directory_iterator(shard_dir)) {
        if (entry.path().extension() == ".work") {
            std::ifstream in(entry.path(), std::ios::binary);
            PackedState s;
            while (in.read(reinterpret_cast<char*>(&s), sizeof(PackedState))) {
                h_work_queue.push_back(s);
            }
            processed_files.push_back(entry.path());
        }
    }

    if (h_work_queue.empty()) {
        std::cout << "Shard " << shard_dir << ": No work to process.\n";
        return 0;
    }
    size_t num_states = h_work_queue.size();
    std::cout << "Shard " << shard_dir << ": Processing " << num_states << " states from " << processed_files.size() << " file(s)...\n";

    auto h_visited = std::make_unique<uint8_t[]>(BITSET_BYTES);
    std::string bitset_file = shard_dir + "/visited.bitset";
    std::ifstream in_bitset(bitset_file, std::ios::binary);
    if(in_bitset) in_bitset.read(reinterpret_cast<char*>(h_visited.get()), BITSET_BYTES);

    uint8_t* d_visited;
    PackedState* d_work_queue;
    uint32_t* d_overflow_counter;
    uint32_t h_overflow_counter = 0;
    CUDA_CHECK(cudaMalloc(&d_visited, BITSET_BYTES));
    CUDA_CHECK(cudaMalloc(&d_work_queue, num_states * sizeof(PackedState)));
    CUDA_CHECK(cudaMalloc(&d_overflow_counter, sizeof(uint32_t)));

    std::vector<PackedState*> h_buckets(NUM_SHARDS, nullptr);
    PackedState** d_buckets;
    uint32_t* h_bucket_counters = new uint32_t[NUM_SHARDS]();
    uint32_t* d_bucket_counters;

    size_t max_single_bucket_size = num_states; 
    for (size_t i = 0; i < NUM_SHARDS; ++i) {
        CUDA_CHECK(cudaMalloc(&h_buckets[i], max_single_bucket_size * sizeof(PackedState)));
    }
    CUDA_CHECK(cudaMalloc(&d_buckets, NUM_SHARDS * sizeof(PackedState*)));
    CUDA_CHECK(cudaMalloc(&d_bucket_counters, NUM_SHARDS * sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMemcpy(d_visited, h_visited.get(), BITSET_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_work_queue, h_work_queue.data(), num_states * sizeof(PackedState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_buckets, h_buckets.data(), NUM_SHARDS * sizeof(PackedState*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bucket_counters, h_bucket_counters, NUM_SHARDS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_overflow_counter, &h_overflow_counter, sizeof(uint32_t), cudaMemcpyHostToDevice));

    size_t grid_size = (num_states + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    process_states_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_work_queue, num_states, d_visited, d_buckets, d_bucket_counters, max_single_bucket_size, d_overflow_counter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_overflow_counter, d_overflow_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (h_overflow_counter > 0) {
        std::cerr << "!!! FATAL ERROR: " << h_overflow_counter << " states were dropped due to bucket overflow. !!!\n";
    }

    CUDA_CHECK(cudaMemcpy(h_visited.get(), d_visited, BITSET_BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bucket_counters, d_bucket_counters, NUM_SHARDS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < NUM_SHARDS; ++i) {
        uint32_t count = h_bucket_counters[i];
        if (count > 0) {
            std::vector<PackedState> h_output_buffer(count);
            CUDA_CHECK(cudaMemcpy(h_output_buffer.data(), h_buckets[i], count * sizeof(PackedState), cudaMemcpyDeviceToHost));
            
            char target_shard_name[4];
            snprintf(target_shard_name, sizeof(target_shard_name), "%02zx", i);
            std::filesystem::path target_dir = std::filesystem::path(shard_dir).parent_path() / target_shard_name;
            std::filesystem::create_directories(target_dir);

            // Using shard_dir in the filename helps trace work origin
            std::string batch_filename = "batch_" + shard_dir + "_" + std::to_string(time(nullptr)) + ".work";
            std::ofstream out(target_dir / batch_filename, std::ios::binary);
            out.write(reinterpret_cast<const char*>(h_output_buffer.data()), count * sizeof(PackedState));
        }
    }
    
    // Atomically save the updated bitset
    std::string tmp_bitset = bitset_file + ".tmp";
    std::ofstream out_bitset(tmp_bitset, std::ios::binary | std::ios::trunc);
    out_bitset.write(reinterpret_cast<const char*>(h_visited.get()), BITSET_BYTES);
    out_bitset.close();
    std::filesystem::rename(tmp_bitset, bitset_file);
    
    // NEW: Rename processed files for safety, then delete at the very end.
    for (const auto& p : processed_files) {
        try {
             std::filesystem::rename(p, p.string() + ".processed");
        } catch (const std::filesystem::filesystem_error& e) {
             std::cerr << "Warning: could not rename " << p << ": " << e.what() << std::endl;
        }
    }
    
    // Cleanup
    for (size_t i = 0; i < NUM_SHARDS; ++i) cudaFree(h_buckets[i]);
    cudaFree(d_buckets);
    cudaFree(d_bucket_counters);
    cudaFree(d_visited);
    cudaFree(d_work_queue);
    cudaFree(d_overflow_counter);
    delete[] h_bucket_counters; // <-- FIXED: Was a memory leak

    std::cout << "Shard " << shard_dir << ": Cleanup complete.\n";
    return 0;
}
