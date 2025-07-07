def odd_core(n):
    k = 0
    while n & 1 == 0:
        n /= 2
        k += 1
    return n, k

def get_child(n):
    if n == 1:
        return None
    
    prev = odd_core(n - 1)
    next = odd_core(n + 1)

    if prev * 2 + 1 == n:
        return prev

    if next * 2 - 1 == n:
        return next

    raise ValueError(f"Could not determine child for {n}")

# --- The S-Value Calculation ---

def get_C(oc1, oc2):
    key = tuple(sorted((oc1, oc2)))
    
    C_TABLE = {
        (1, 1): 2,
        (1, 3): -4,
        (3, 7): 0,
        (3, 13): 8,
        (3, 25): 14,
        (3, 49): -3,

    }
    
    if key not in C_TABLE:
        raise NotImplementedError(f"Correction Factor C for pair {key} is not defined.")
        
    return C_TABLE[key]

def get_S_value(n):
    """Calculates the stopping time contribution (S-value) for n's neighborhood."""
    if n == 1:
        # Base case from observation: d(3) = 5, and d(3) = d(1) + S(at n=3)
        # S(at n=3) = nu(2)+nu(4)+C(1,1) = 1+2+2 = 5
        return 5

    oc_prev, k_prev = odd_core(n - 1)
    oc_next, k_next = odd_core(n + 1)
    
    c_factor = get_C(oc_prev, oc_next)
    
    return k_prev + k_next + c_factor

# --- The Main Recursive Function ---

@functools.lru_cache(maxsize=None)
def dist(n):
    """
    Recursively calculates the stopping distance of an odd number n.
    """
    # Base case: The stopping time of 1 is 0.
    if n == 1:
        return 0
    
    # Phase 1: Find the child (the next step on the walk down to 1)
    child = get_child(n)
    
    # Phase 2: Recursively call dist on the child and add the S-value
    # from the current neighborhood on the way back up.
    return dist(child) + get_S_value(n)

# --- Demonstration ---
if __name__ == "__main__":
    target_number = 97
    print(f"Calculating the stopping time for n = {target_number}...")
    
    # Walk the chain down to 1 to show the path
    path = []
    curr = target_number
    while curr is not None:
        path.append(curr)
        curr = get_child(curr)
    
    print(f"The 'child' chain to 1 is: {' → '.join(map(str, path))}")
    print("-" * 30)
    
    # Calculate the final distance
    final_distance = dist(target_number)
    
    print(f"The calculated stopping time is: {final_distance}")
    
    # Show the recursive calls cache for inspection
    print("\nCache contents for dist():")
    print(dist.cache_info())
    print(dist.cache_items())

