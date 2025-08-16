# automaton.py - Collatz odd-to-odd map as a counter machine.
# Stuart Riffle
#
# This is a one-register automaton that performs the Collatz transform without performing
# arithmetic on n. It carries the 2-adic valuations (odd core and k) of n's immediate 
# even neighbors nl to the left (n-1), and nr to the right (n+1), as state
#
# Under construction!

TERMINATOR_AT_ONE = (0, 0, 1, 1) 

def low_1s(n):
    k = 0
    while n & (1 << k):
        k += 1
    return k

def low_0s(n):
    return low_1s(~n) if n else 0

def odd_val(n):
    k = low_0s(n)
    oc = n >> k
    return oc, k

def odd_core(n):
    oc, _ = odd_val(n)
    return oc

def is_odd(n):
    return n & 1

def collatz_odd_next(n):
    assert(is_odd(n))
    return odd_core(3 * n + 1)

def collatz_odd_dist(n):
    dist = 0
    while n > 1:
        n = collatz_odd_next(n)
        dist += 1
    return dist

def create_automaton(n):
    ocl, kl = odd_val(n - 1)
    ocr, kr = odd_val(n + 1)
    return (ocl, kl, ocr, kr)

def validate_automaton(t):
    assert(t)
    assert(len(t) == 4)
    if t == TERMINATOR_AT_ONE:
        return

    ocl, kl, ocr, kr = t
    nl = ocl << kl
    nr = ocr << kr

    assert(ocl > 0 and ocr > 0)
    assert(is_odd(ocl) and is_odd(ocr))
    assert(min(kl, kr) == 1)
    assert(max(kl, kr) > 1)
    assert(nl + 1 == nr - 1)

def peek_automaton(t):
    ocl, kl, _, _ = t
    n = (ocl << kl) + 1
    return n 

# This is the orbit of n=27 as an example.
#
# The automaton has 3 or 4 basic transitions. Cases A and B involve (roughly) tripling nl and nr respectively.
# They can be identified by the values of kl and kr.
# 
#  dist,       nl, ofactorsl,        n, factors,           nr, ofactorsr, case
#    41,   (13 1), {13},            27, {3},            (7 2), {7},          B
#    40,    (5 3), {5},             41, {41},          (21 1), {3 7},        A
#    39,   (15 1), {3 5},           31, {31},           (1 5), {},           B
#    38,   (23 1), {23},            47, {47},           (3 4), {3},          B
#    37,   (35 1), {5 7},           71, {71},           (9 3), {3},          B
#    36,   (53 1), {53},           107, {107},         (27 2), {3},          B
#    35,    (5 5), {5},            161, {7 23},        (81 1), {3},          A
#    34,   (15 3), {3 5},          121, {11},          (61 1), {61},         A
#    33,   (45 1), {3 5},           91, {7 13},        (23 2), {23},         B
#    32,   (17 3), {17},           137, {137},         (69 1), {3 23},       A
#    31,   (51 1), {3 17},         103, {103},         (13 3), {13},         B
#    30,   (77 1), {7 11},         155, {5 31},        (39 2), {3 13},       B
#    29,   (29 3), {29},           233, {233},        (117 1), {3 13},       A
#    28,   (87 1), {3 29},         175, {5 7},         (11 4), {11},         B
#    27,  (131 1), {131},          263, {263},         (33 3), {3 11},       B
#    26,  (197 1), {197},          395, {5 79},        (99 2), {3 11},       B
#    25,   (37 4), {37},           593, {593},        (297 1), {3 11},       A
#    24,  (111 2), {3 37},         445, {5 89},       (223 1), {223},        -
#    23,   (83 1), {83},           167, {167},         (21 3), {3 7},        B
#    22,  (125 1), {5},            251, {251},         (63 2), {3 7},        B
#    21,   (47 3), {47},           377, {13 29},      (189 1), {3 7},        A
#    20,  (141 1), {3 47},         283, {283},         (71 2), {71},         B
#    19,   (53 3), {53},           425, {5 17},       (213 1), {3 71},       A
#    18,  (159 1), {3 53},         319, {11 29},        (5 6), {5},          B
#    17,  (239 1), {239},          479, {479},         (15 5), {3 5},        B
#    16,  (359 1), {359},          719, {719},         (45 4), {3 5},        B
#    15,  (539 1), {7 11},        1079, {13 83},      (135 3), {3 5},        B
#    14,  (809 1), {809},         1619, {1619},       (405 2), {3 5},        B
#    13,  (607 2), {607},         2429, {7 347},     (1215 1), {3 5},        -
#    12,  (455 1), {5 7 13},       911, {911},         (57 4), {3 19},       B
#    11,  (683 1), {683},         1367, {1367},       (171 3), {3 19},       B
#    10, (1025 1), {5 41},        2051, {7 293},      (513 2), {3 19},       B
#     9,  (769 2), {769},         3077, {17 181},    (1539 1), {3 19},       -
#     8,    (9 6), {3},            577, {577},        (289 1), {17},         A
#     7,   (27 4), {3},            433, {433},        (217 1), {7 31},       A
#     6,   (81 2), {3},            325, {5 13},       (163 1), {163},        -
#     5,   (15 2), {3 5},           61, {61},          (31 1), {31},         -
#     4,   (11 1), {11},            23, {23},           (3 3), {3},          B
#     3,   (17 1), {17},            35, {5 7},          (9 2), {3},          B
#     2,   (13 2), {13},            53, {53},          (27 1), {3},          -
#     1,    (1 2), {},               5, {5},            (3 1), {3},          -
#     0,    (0 0), {},               1, {},             (1 1), {},           -
# 
# The remaining case is less clear. Instead of operating (effectively) on nl or nr,
# the value of n is set to about 3/4 of one of them. If it's nr, it seems like n'
# is _also_ roughly 1.5 ocl. That may be a way to tell the two cases apart.
# "Control" (meaning k > 1) is given to one side based on this.
#
# dist
#  24:  n' = ocr (223)  * 0.75 ish = 167   (or ocl * 1.5 ish)  giving control right
#  13:  n' = ocr (1215) * 0.75 ish = 911   (or ocl * 1.5 ish)  giving control right
#   9:  n' = ocl (769)  * 0.75 ish = 577                       giving control left
#   6:  n' = ocl (81)   * 0.75 ish = 61                        giving control left
#   5:  n' = ocr (31)   * 0.75 ish = 23    (or ocl * 1.5 ish)  giving control right
#   2:  ?
#
# The goal is an automaton that avoids 2-adic re-valuation to the extent possible,
# and that runs without "cheating" and reconstructing n. At any point it's easy
# to calculate nl, add 1, do Collatz, and subtract 1 again. And there are a million
# ways to obfuscate it. But the idea is to avoid doing that entirely, even as 
# a temporary value in a larger expression, to demonstrate that we're running on
# basic arithmetic relationships, not some chaotic magic inside n itself.
#
# The A and B cases are currently forced to re-value the other sides. In most cases
# the result isn't even used on the next step! There must be some way around this.

def step_automaton(t):
    (ocl, kl, ocr, kr) = t

    if t == TERMINATOR_AT_ONE:
        return t

    if kl > 2 and kr == 1:
        # CASE A
        ocl, kl = 3 * ocl, kl - 2
        ocr, kr = odd_val((3 * ocr + 1) >> 1) # FIXME - recalculate directly (without reconstructing n)
        return (ocl, kl, ocr, kr)

    if kr > 1 and kl == 1:
        # CASE B
        ocr, kr = 3 * ocr, kr - 1
        ocl, kl = odd_val(3 * ocl + 1) # FIXME - recalculate directly (without reconstructing n)
        return (ocl, kl, ocr, kr)

    assert(kl == 2 and kr == 1)

    # CASE C (?)

    # The low bits in 3*ocl predict the k drop of 3n+1

    m  = 3 * ocl
    k  = low_1s(m)
    m -= (1 << k)
    nl = (m >> k) + 1
    kl = 2 - ((nl >> 1) & 1)
    kr = 3 - kl

    return (nl >> kl, kl, (nl + 2) >> kr, kr)

def run_automaton(n):
    t = create_automaton(n)
    while t != TERMINATOR_AT_ONE:
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
        

                                                                                    


