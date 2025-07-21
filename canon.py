import argparse
arg = argparse.ArgumentParser()
arg.add_argument("--min", type=int, default=1)
arg.add_argument("--max", type=int, default=100)
arg.add_argument("--odds", action="store_true")
arg = arg.parse_args()


def is_pow_two(n):
    return n & (n - 1) == 0

def padic(base, n):
    k = 0
    if n != 0:
        while n % base == 0:
            n = n // base
            k += 1
    return n, k

def odd_core(n):
    return padic(2, n)

def collatz_odd_next(n):
    n = 3 * n + 1
    while n % 2 == 0:
        n = n // 2
    return n

def collatz_odd_dist(n):
    dist = 0
    while n > 1:
        dist += 1
        n = collatz_odd_next(n)
    return dist

def seed_to_root_dist(n, indent = 0):
    # Handle trivial cases
    if is_pow_two(n):
        return 0
    if is_pow_two(n * 3 + 1):
        return 1
    
    # Find the neighbor with the lowest k
    oc_l, k_l = odd_core(n - 1)
    oc_r, k_r = odd_core(n + 1)
    
    oc_lo, k_lo, oc_hi, k_hi = (oc_l, k_l, oc_r, k_r) if k_l <= k_r else (oc_r, k_r, oc_l, k_l)

    # Follow its odd core to the root of the tree
    dist = seed_to_root_dist(oc_lo, indent + 4)
    actual = collatz_odd_dist(n)
    error = dist - actual

    print(f"{' ' * indent}n = {n}, lo {oc_lo}, k={k_lo}, hi {oc_hi}, k={k_hi}, dist({oc_lo}) was calculated as {dist} (should be {actual}, error {error})")

    # Every level appears to have a fixed cost
    level_cost = 6 # FIXME
    dist += level_cost

    # Add the variable contribution of this level
    dist += k_hi # FIXME

    return dist

def reduce_to_seed(n):
    # Return the distance class seed, or n if this is not a doubling
    return n # FIXME

def predict_odd_dist(n):
    seed = reduce_to_seed(n)
    print(f"Predicting {n} seed = {seed}")
    return seed_to_root_dist(seed)


def predict_odd_dist_2(n):
    if n == 1:
        return 0
    
    # Find the neighbor with the lowest k

    oc_l,  k_l  = odd_core(n - 1)
    oc_r,  k_r  = odd_core(n + 1)
    oc_lo, k_lo = (oc_l, k_l) if k_l <= k_r else (oc_r, k_r)

    dist = predict_odd_dist_2(oc_lo)

    if k_lo > 1:
        dist += 6

    if n <= 15:
        dist -= 1

    return dist



def booktable():
    cols = 5
    rows = 100
    pages = 1
    

def dataset():
    # --min <num>=1 --range <num>=100 --odds 

    for n in range(arg.min, arg.max + 1):
        if arg.odds and n % 2 == 0:
            continue

        dist        = collatz_odd_dist(n)
        predicted   = predict_odd_dist_2(n)

        print(f"{n:-3}: {dist:-3}, {predicted:-3}, error {predicted - dist}")



if __name__ == "__main__":

    # commands
    # canon.py booktable
    # canon.py dataset

    if arg.command == "booktable":
        booktable()
    elif arg.command == "dataset":
        dataset()


    main()
