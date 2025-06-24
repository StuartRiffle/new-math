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

// Collatz Pipeline BFS with Asynchronous I/O and GPU Processing
// Accepts a list of shard work folders, processes them in an overlapped 3-stage pipeline

// This program performs a breadth first traversal of the Collatz state space. //
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
// requires one bit for each of them, so we'll need almost 78 TB of memory (uncompressed)
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
// of the state space. Each step "scatters" all the active orbits to a different shard.
// The space is partitioned to balance this outgoing traffic between the other shards.
// Incoming batches are stored in their own files, which amounts to a simple work queue.
// The frontier of the search is pushed around between shards in this way. The shards can
// be processed serially or in parallel, because all persistent state is stored in files.
//
// This process is heavily I/O bound. The computation itself is trivial, but the bitset
// is huge, and "message passing" of states involves a lot of bandwidth. This is a
// gather/scatter operation that must be performed thousands of times.
//
// Breathe in, breathe out.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <future>
#include <array>
#include <chrono>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>

// ==== CUDA Headers and Error Helper ====
#include <cuda_runtime.h>
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// ==== Device Constants and Helpers ====
constexpr size_t SMALL_PRIMES_PRODUCT = 3 * 5 * 7 * 11 * 13;
__constant__ uint32_t D_MODULI[]      = { SMALL_PRIMES_PRODUCT, 17, 19, 23, 29, 31, 37, 41, 43, 47 };
__constant__ uint32_t D_FIELD_BITS[]  = { 15, 5, 5, 5, 5, 5, 6, 6, 6, 6 };
__constant__ uint64_t D_FIELD_MASK[]  = { 0x7FFF, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x3F, 0x3F, 0x3F, 0x3F };

constexpr int NUM_FIELDS = 10;
using PackedState = uint64_t;

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

__device__ size_t state_to_shard_id(PackedState s, size_t SHARD_BITS) {
    return s >> (64 - SHARD_BITS);
}

__device__ size_t state_to_idx_in_shard(PackedState s, size_t SHARD_SIZE) {
    return s & (SHARD_SIZE - 1);
}

// ==== Device Kernel ====
__global__ void process_states_kernel(
    const PackedState* work_queue,
    size_t num_states,
    uint8_t* visited_bitset,
    PackedState** outgoing_buckets,
    uint32_t* bucket_counters,
    size_t max_bucket_size,
    uint32_t* overflow_counter,
    size_t SHARD_BITS,
    size_t SHARD_SIZE,
    size_t NUM_SHARDS
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    PackedState s = work_queue[tid];
    size_t idx = state_to_idx_in_shard(s, SHARD_SIZE);
    uint8_t mask = 1 << (idx % 8);
    uint8_t old_val = atomicOr(&visited_bitset[idx / 8], mask);

    if ((old_val & mask) != 0) return;

    std::array<uint16_t, NUM_FIELDS> residues = unpack_residues(s);
    residues = collatz_residue_step(residues);
    PackedState neighbor = pack_residues(residues);

    size_t target_shard = state_to_shard_id(neighbor, SHARD_BITS);
    uint32_t append_idx = atomicAdd(&bucket_counters[target_shard], 1);

    if (append_idx < max_bucket_size) {
        outgoing_buckets[target_shard][append_idx] = neighbor;
    } else {
        atomicAdd(overflow_counter, 1);
    }
}

// ==== Shard Pipeline Structs ====
using ShardID = size_t;
struct ShardParams {
    size_t SHARD_BITS = 8;
    size_t NUM_SHARDS = 1ULL << 8;
    size_t SHARD_SIZE = 1ULL << 20;
    size_t BITSET_BYTES = (1ULL << 20) / 8;
};

struct ShardBuffer {
    std::vector<PackedState> work_queue;
    std::unique_ptr<uint8_t[]> bitset;
    std::vector<std::filesystem::path> processed_files;
    std::string shard_dir;
    std::vector<std::vector<PackedState>> outgoing_buckets;
    std::vector<uint32_t> bucket_counters;
    uint32_t overflow_counter = 0;
    size_t num_states = 0;
};

// ==== CLI Argument Parsing ====
ShardParams parse_args(int argc, char* argv[], std::vector<std::string>& shard_dirs) {
    ShardParams params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--shard-bits" && i+1 < argc) {
            params.SHARD_BITS = std::stoi(argv[++i]);
            params.NUM_SHARDS = 1ULL << params.SHARD_BITS;
        } else if (arg == "--num-shards" && i+1 < argc) {
            params.NUM_SHARDS = std::stoull(argv[++i]);
            size_t bits = 0; size_t n = params.NUM_SHARDS;
            while (n > 1) { n >>= 1; bits++; }
            params.SHARD_BITS = bits;
        } else if (arg == "--shard-size" && i+1 < argc) {
            params.SHARD_SIZE = std::stoull(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: program [--shard-bits N | --num-shards N | --shard-size N] <shard_dir_1> ...\n";
            exit(0);
        } else {
            shard_dirs.push_back(arg);
        }
    }
    params.BITSET_BYTES = params.SHARD_SIZE / 8;
    return params;
}

// ==== Async Loader ====
void load_shard_async(const ShardParams& params, const std::string& shard_dir, ShardBuffer* buf) {
    buf->work_queue.clear();
    buf->processed_files.clear();
    buf->shard_dir = shard_dir;
    buf->outgoing_buckets.clear();
    buf->bucket_counters.assign(params.NUM_SHARDS, 0);
    buf->overflow_counter = 0;
    buf->outgoing_buckets.resize(params.NUM_SHARDS);

    for (const auto& entry : std::filesystem::directory_iterator(shard_dir)) {
        if (entry.path().extension() == ".work") {
            std::ifstream in(entry.path(), std::ios::binary);
            PackedState s;
            while (in.read(reinterpret_cast<char*>(&s), sizeof(PackedState)))
                buf->work_queue.push_back(s);
            buf->processed_files.push_back(entry.path());
        }
    }
    buf->num_states = buf->work_queue.size();
    buf->bitset = std::make_unique<uint8_t[]>(params.BITSET_BYTES);
    std::fill(buf->bitset.get(), buf->bitset.get() + params.BITSET_BYTES, 0);
    std::string bitset_file = shard_dir + "/visited.bitset";
    std::ifstream in_bitset(bitset_file, std::ios::binary);
    if (in_bitset) {
        in_bitset.read(reinterpret_cast<char*>(buf->bitset.get()), params.BITSET_BYTES);
    }
}

// ==== Async Writer ====
void write_shard_async(const ShardParams& params, ShardBuffer* buf) {
    std::string bitset_file = buf->shard_dir + "/visited.bitset";
    std::string tmp_bitset = bitset_file + ".tmp";
    std::ofstream out_bitset(tmp_bitset, std::ios::binary | std::ios::trunc);
    out_bitset.write(reinterpret_cast<const char*>(buf->bitset.get()), params.BITSET_BYTES);
    out_bitset.close();
    std::filesystem::rename(tmp_bitset, bitset_file);

    for (size_t i = 0; i < params.NUM_SHARDS; ++i) {
        uint32_t count = buf->bucket_counters[i];
        if (count > 0) {
            const auto& out_buf = buf->outgoing_buckets[i];
            char target_shard_name[8];
            snprintf(target_shard_name, sizeof(target_shard_name), "%0*zx", int((params.SHARD_BITS+3)/4), int(i));
            std::filesystem::path target_dir = std::filesystem::path(buf->shard_dir).parent_path() / target_shard_name;
            std::filesystem::create_directories(target_dir);
            std::string batch_filename = "batch_" + buf->shard_dir + "_" + std::to_string(std::time(nullptr)) + ".work";
            std::ofstream out(target_dir / batch_filename, std::ios::binary);
            out.write(reinterpret_cast<const char*>(out_buf.data()), count * sizeof(PackedState));
        }
    }

    for (const auto& p : buf->processed_files) {
        try {
            std::filesystem::rename(p, p.string() + ".processed");
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Warning: could not rename " << p << ": " << e.what() << std::endl;
        }
    }
}

// ==== GPU Processing ====
// All CUDA API calls are made from the main thread ONLY
void process_shard_gpu(const ShardParams& params, ShardBuffer& buf) {
    if (buf.num_states == 0) return;

    uint8_t* d_visited;
    PackedState* d_work_queue;
    CUDA_CHECK(cudaMalloc(&d_visited, params.BITSET_BYTES));
    CUDA_CHECK(cudaMalloc(&d_work_queue, buf.num_states * sizeof(PackedState)));
    CUDA_CHECK(cudaMemcpy(d_visited, buf.bitset.get(), params.BITSET_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_work_queue, buf.work_queue.data(), buf.num_states * sizeof(PackedState), cudaMemcpyHostToDevice));

    std::vector<PackedState*> d_buckets(params.NUM_SHARDS, nullptr);
    for (size_t i = 0; i < params.NUM_SHARDS; ++i) {
        CUDA_CHECK(cudaMalloc(&d_buckets[i], buf.num_states * sizeof(PackedState)));
    }
    PackedState** d_buckets_ptr;
    CUDA_CHECK(cudaMalloc(&d_buckets_ptr, params.NUM_SHARDS * sizeof(PackedState*)));
    CUDA_CHECK(cudaMemcpy(d_buckets_ptr, d_buckets.data(), params.NUM_SHARDS * sizeof(PackedState*), cudaMemcpyHostToDevice));

    uint32_t* d_bucket_counters;
    CUDA_CHECK(cudaMalloc(&d_bucket_counters, params.NUM_SHARDS * sizeof(uint32_t)));
    std::vector<uint32_t> h_bucket_counters(params.NUM_SHARDS, 0);
    CUDA_CHECK(cudaMemcpy(d_bucket_counters, h_bucket_counters.data(), params.NUM_SHARDS * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t* d_overflow_counter;
    uint32_t h_overflow_counter = 0;
    CUDA_CHECK(cudaMalloc(&d_overflow_counter, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_overflow_counter, &h_overflow_counter, sizeof(uint32_t), cudaMemcpyHostToDevice));

    constexpr int THREADS_PER_BLOCK = 256;
    size_t grid_size = (buf.num_states + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    process_states_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_work_queue,
        buf.num_states,
        d_visited,
        d_buckets_ptr,
        d_bucket_counters,
        buf.num_states,
        d_overflow_counter,
        params.SHARD_BITS,
        params.SHARD_SIZE,
        params.NUM_SHARDS
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(buf.bitset.get(), d_visited, params.BITSET_BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bucket_counters.data(), d_bucket_counters, params.NUM_SHARDS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_overflow_counter, d_overflow_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (h_overflow_counter > 0) {
        std::cerr << "!!! FATAL ERROR: " << h_overflow_counter << " states dropped due to bucket overflow in " << buf.shard_dir << std::endl;
    }
    buf.bucket_counters = h_bucket_counters;
    buf.overflow_counter = h_overflow_counter;

    for (size_t i = 0; i < params.NUM_SHARDS; ++i) {
        uint32_t count = h_bucket_counters[i];
        if (count > 0) {
            buf.outgoing_buckets[i].resize(count);
            CUDA_CHECK(cudaMemcpy(buf.outgoing_buckets[i].data(), d_buckets[i], count * sizeof(PackedState), cudaMemcpyDeviceToHost));
        }
    }

    for (size_t i = 0; i < params.NUM_SHARDS; ++i) cudaFree(d_buckets[i]);
    cudaFree(d_buckets_ptr);
    cudaFree(d_bucket_counters);
    cudaFree(d_visited);
    cudaFree(d_work_queue);
    cudaFree(d_overflow_counter);
}

// ==== Main Pipeline Loop ====
int main(int argc, char* argv[]) {
    std::vector<std::string> shard_dirs;
    ShardParams params = parse_args(argc, argv, shard_dirs);

    if (shard_dirs.empty()) {
        std::cerr << "Usage: " << argv[0] << " [--shard-bits N | --num-shards N | --shard-size N] <shard_dir_1> <shard_dir_2> ...\n";
        return 1;
    }

    constexpr int PIPELINE_DEPTH = 3;
    std::array<ShardBuffer, PIPELINE_DEPTH> pipeline_bufs;
    std::array<std::future<void>, PIPELINE_DEPTH> io_futures;

    size_t num_shards = shard_dirs.size();

    // Start the first load
    if (num_shards > 0)
        io_futures[0] = std::async(std::launch::async, load_shard_async, std::ref(params), shard_dirs[0], &pipeline_bufs[0]);

    for (size_t i = 0; i < num_shards + PIPELINE_DEPTH - 1; ++i) {
        size_t buf_idx = i % PIPELINE_DEPTH;

        // Wait for previous IO on this buffer (load or store) to finish
        if (i >= PIPELINE_DEPTH - 1)
            io_futures[buf_idx].wait();

        // Process this buffer on GPU after loading
        if (i < num_shards) {
            io_futures[buf_idx].wait();
            std::cout << "[Shard " << pipeline_bufs[buf_idx].shard_dir << "] Loaded, num_states: " << pipeline_bufs[buf_idx].num_states << "\n";
            process_shard_gpu(params, pipeline_bufs[buf_idx]);
            io_futures[buf_idx] = std::async(std::launch::async, write_shard_async, std::ref(params), &pipeline_bufs[buf_idx]);
            std::cout << "[Shard " << pipeline_bufs[buf_idx].shard_dir << "] Processed, async scatter started.\n";
        }

        // Start loading the next shard if any
        size_t next_shard = i + PIPELINE_DEPTH;
        if (next_shard < num_shards + PIPELINE_DEPTH - 1 && next_shard - (PIPELINE_DEPTH - 1) < num_shards) {
            size_t next_buf_idx = (i + 1) % PIPELINE_DEPTH;
            size_t next_shard_idx = next_shard - (PIPELINE_DEPTH - 1);
            if (next_shard_idx < num_shards) {
                io_futures[next_buf_idx] = std::async(std::launch::async, load_shard_async, std::ref(params), shard_dirs[next_shard_idx], &pipeline_bufs[next_buf_idx]);
            }
        }
    }

    std::cout << "All shards processed. Exiting.\n";
    return 0;
}



















