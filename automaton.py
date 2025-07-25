# automaton.py - Collatz odd-to-odd map as a "blind" counter machine.
# Stuart Riffle
#
# This is a one-register automaton that performs the Collatz transform without performing
# arithmetic on n. It carries the 2-adic valuations (odd core and k) of n's immediate 
# even neighbors n-1 and n+1, which completely determine the evolution of n itself. That
# 4-tuple of integers can model the Collatz map without simulating the binary avalanche.

TERMINAL_STATE = (0, 0, 1, 1) # 1

def v2(n):
    k = 0
    while n and n & (1 << k) == 0:
        k += 1
    return k

def odd_core(n):
    k = v2(n)
    return n >> k, k

def create_automaton(n):
    ocl, kl = odd_core(n - 1)
    ocr, kr = odd_core(n + 1)
    return (ocl, kl, ocr, kr)

def step_automaton(t):
    if t == TERMINAL_STATE:
        return t

    ocl, kl, ocr, kr = t

    if kl >= 3 and kr == 1:
        ocl, kl = 3 * ocl, kl - 2
        ocr, kr = ocl * (1 << (kl - 1)) + 1, 1

    elif kr >= 2 and kl == 1:
        ocr, kr = 3 * ocr, kr - 1
        ocl, kl = ocr * (1 << (kr - 1)) - 1, 1

    else: 
        assert(kl == 2 and kr == 1)

        # A trick: we predict the next k drop without performing +1 and measuring the avalanche,
        # by sampling the low *set* bits of ocl*3 (because we know +1 will flip exactly these).
        # Looping k to count bits makes this a "counter machine" instead of a pure transducer
        # (but still not Turing complete).

        k = 0
        m = 3 * ocl
        while m & (1 << k):
            k += 1

        # The extracted k value now lets us calculate the next state tuple in closed form, 
        # withoutmore looping to recalculate neighbor 2-adic valuations.

        nl = ((m + 1) >> k) - 1   # n' - 1
        nr = nl + 2               # n' + 1

        kl, kr = (2, 1) if nl % 4 == 0 else (1, 2)
        ocl, ocr = nl >> kl, nr >> kr

    return ocl, kl, ocr, kr

def peek_automaton(t):
    ocl, kl, _, _ = t
    return (ocl << kl) + 1

def validate_automaton(t):
    if t == TERMINAL_STATE:
        return

    assert(len(t) == 4)
    assert(all(isinstance(x, int) for x in t))
    
    ocl, kl, ocr, kr = t
    assert(ocl & 1 == 1)
    assert(ocr & 1 == 1)
    assert(kl == 1 or kr == 1)

    nl = (ocl << kl) + 1
    nr = (ocr << kr) - 1
    assert(nl == nr)
    assert(nl & 1 == 1)


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

def run_automaton(n):
    t = create_automaton(n)
    while t != TERMINAL_STATE:
        print(f"{t} [{peek_automaton(t)}]")
        t = step_automaton(t)
    return t

def test_orbit(n):
    while n != 1:
        t = create_automaton(n)
        tt = step_automaton(t)
        next = collatz_odd_next(n)
        print(f'{n:3} -> {next:3}  {t} -> {tt}')

        n = next

if __name__ == "__main__":
    test_orbit(27)
        