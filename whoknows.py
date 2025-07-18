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

def is_pow_two(n):
    return n & (n - 1) == 0


def calc_stopping_distance(n):
    if is_pow_two(n * 3 + 1):
        return 1

    steps = 4
    fixup = 0

    while n > 1:
        oc_l, k_l



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

