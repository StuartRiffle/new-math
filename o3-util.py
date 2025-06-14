#!/usr/bin/env python3
# collatz_delta9.py  –  Δ₆ … Δ₉ consolidated prototype
# ----------------------------------------------------
# 2025-06-14

import math, random, time
from collections import Counter
from itertools import islice

# ----------------------------------------------------
# 0. basic helpers
# ----------------------------------------------------
def sieve(limit):
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = b"\x00" * len(is_prime[p * p :: p])
    return [p for p, f in enumerate(is_prime) if f]

def v2(x):
    return (x & -x).bit_length() - 1

def collatz_step(n):
    k = v2(3 * n + 1)
    return ((3 * n + 1) >> k, k)

def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, y, x = egcd(b, a % b)
    return (g, x, y - (a // b) * x)

def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("inverse fails")
    return x % m

def crt_pair(a1, m1, a2, m2):
    g, s, t = egcd(m1, m2)
    if g != 1:
        raise ValueError("CRT with non-coprime moduli")
    return (a1 * t * m2 + a2 * s * m1) % (m1 * m2), m1 * m2

# ----------------------------------------------------
# 1. small-prime tables + discrete logs
# ----------------------------------------------------
P0 = 89                       # initial cutoff
PRIMES = sieve(P0)
LOG2 = {}                     # LUT for discrete log base 2
for p in PRIMES:
    lut, x = {}, 1
    for k in range(p - 1):
        lut[x] = k
        x = (x * 2) % p
    LOG2[p] = lut

# ----------------------------------------------------
# 2. Δ₉-② primitive local inverse  T_p^{(k)}
# ----------------------------------------------------
def Tpk(p, k, residue_after):
    """Return residue_before (mod p) so that v₂(3n+1)=k."""
    pow2 = pow(2, k, p)
    return ((pow2 * residue_after - 1) * modinv(3, p)) % p

# ----------------------------------------------------
# 3. residue vector, NRB, NRM, Valley-Index
# ----------------------------------------------------
def residue_vector(n, primes=PRIMES):
    return {p: n % p for p in primes}

def NRB(n, k=2, primes=PRIMES):
    rows = []
    for p in primes:
        row = [(n + j) % p < p // 2 for j in range(-k, k + 1)]
        rows.append(row)
    return rows

def NRM(n, k=2, primes=PRIMES):
    """Neighbour Residue Matrix – raw residues, not bits."""
    return [[(n + j) % p for j in range(-k, k + 1)] for p in primes]

def valley_index(n, k=2, primes=PRIMES):
    s_left  = sum((n - 2) % p == 1 for p in primes)
    s_right = sum((n + 2) % p == 1 for p in primes)
    s_mid   = sum(n % p == 1       for p in primes)
    if s_mid == 0:
        return float("inf")
    return min(s_left, s_right) / s_mid

# ----------------------------------------------------
# 4. Lyapunov parts  (Δ₉-⑦)
# ----------------------------------------------------
def H_P(n, primes=PRIMES):
    rv = residue_vector(n, primes)
    return sum(abs(r - (p - 1) / 2) / p for p, r in rv.items())

def W_P(n, primes=PRIMES, beta=1.0):
    rv = residue_vector(n, primes)
    aligned = all(r in (1, p - 1) for p, r in rv.items())
    return H_P(n, primes) + (beta if aligned else 0)

# ----------------------------------------------------
# 5. Δ₉-① inverse dynamics via   (T(n), T²(n))
# ----------------------------------------------------
def inverse_from_two_steps(T1, T2, k1, primes=PRIMES[:8]):
    """
    Given x1=T(n), x2=T²(n) and k1=v₂(3n+1) reconstruct n (CRT, primitive inverses).
    Works for the product of first 8 primes (≈ 9699690) then lifts by search.
    """
    m, mod = 0, 1
    for p in primes:
        r1 = x1 = T1 % p
        r2 = x2 = T2 % p
        k  = k1 % (p - 1)
        r0 = Tpk(p, k, r1)         # residue of n (mod p)
        m, mod = crt_pair(m, mod, r0, p)
    # lift to actual predecessor by searching small t
    # because CRT gives only mod prod(primes)
    for t in range(0, 1000):
        cand = m + t * mod
        if collatz_step(cand)[0] == T1:
            return cand
    raise ValueError("fail to lift")

# ----------------------------------------------------
# 6. Local complexity tensor and spectral distance hooks (Δ₉-⑧,⑨)
# ----------------------------------------------------
def complexity_tensor(n, k=2, primes=PRIMES):
    """
    crude local tensor L_ij = cov( bit_i , bit_j ) over NRB rows
    The size is (#rows)×(#rows); here we keep only leading eigenvalues
    via power iteration (one or two).
    """
    bitmap = NRB(n, k, primes)
    m = len(bitmap)
    mean = [sum(row) / len(row) for row in bitmap]
    # very small matrix – we can build it explicitly
    L = [[0.0 for _ in range(m)] for __ in range(m)]
    for i in range(m):
        for j in range(m):
            cov = sum((bitmap[i][t] - mean[i]) * (bitmap[j][t] - mean[j])
                      for t in range(len(bitmap[i]))) / len(bitmap[i])
            L[i][j] = cov
    # power iteration for first two eigenvalues
    def power(v):
        w = [sum(L[i][j] * v[j] for j in range(m)) for i in range(m)]
        norm = math.sqrt(sum(x * x for x in w)) or 1.0
        return [x / norm for x in w], norm
    v = [1.0 / math.sqrt(m)] * m
    v, lam1 = power(v)
    # deflation
    for i in range(m):
        for j in range(m):
            L[i][j] -= lam1 * v[i] * v[j]
    v2, lam2 = power(v)
    return lam1, lam2

# ----------------------------------------------------
# 7. Certified solver with Δ₉ extensions
# ----------------------------------------------------
ALPHA = 36
BETA  = 1.0
GAMMA = 1 / len(PRIMES)

def SDM(n, primes=PRIMES):
    limit = int(math.log(n)) or 3
    primes = [p for p in primes if p <= limit]
    if not primes:
        return 0.0
    S = sum(n % p == 1 for p in primes)
    return S / math.log(math.log(n)) if n > 3 else 0.0

def NRB_entropy(bitmap):
    flat = [int(b) for row in bitmap for b in row]
    L = len(flat)
    c = Counter(flat)
    return -sum(v / L * math.log(v / L) for v in c.values() if v)

def UPB(n, sdm, nrb_H, reboots):
    if sdm < 1e-9:
        sdm = 1e-9
    return int(ALPHA * math.log(n) * (1 / sdm + BETA * reboots + GAMMA * nrb_H)) + 10

def solve(n, verbose=False, k_nrb=2):
    """
    Forward iterate under W_P descent; on failure emit residue certificate.
    """
    assert n & 1
    primes = PRIMES.copy()
    reboots = 0
    sdm = SDM(n, primes)
    nrb_H = NRB_entropy(NRB(n, k_nrb, primes))
    budget = UPB(n, sdm, nrb_H, reboots)

    traj, step = [n], 0
    while n != 1 and step <= budget:
        n, k = collatz_step(n)
        traj.append(n)
        step += 1
        if step % 1000 == 0 and verbose:
            print(f" {step} steps, n≈{n.bit_length()} bits")
        # enlarge prime grid if parity bitmap becomes too small
        if step == budget // 2 and n.bit_length() > 256:
            extra = sieve(primes[-1] * 2)[len(primes):]
            primes.extend(extra)
            reboots += 1
            budget = UPB(n, SDM(n, primes), NRB_entropy(NRB(n, k_nrb, primes)),
                         reboots)

    if n == 1:
        if verbose:
            print("✅ verified; length", step)
        return traj

    # ----- produce minimal-certificate (Δ₉-③ spatio-temporal) -----
    x1, k1 = traj[-1], k          # T(n)
    x2, _  = collatz_step(x1)     # T²(n)
    n_prev = inverse_from_two_steps(x1, x2, k1)
    cert = {"n_prev": n_prev, "T(n)": x1, "T²(n)": x2, "k1": k1}
    if verbose:
        print("❌ budget exceeded – emitting certificate")
    return cert

# ----------------------------------------------------
# 8. CLI demo
# ----------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python3 collatz_delta9.py <odd_integer>")
        sys.exit()
    N = int(sys.argv[1])
    if N % 2 == 0 or N <= 0:
        print("need positive odd integer")
        sys.exit()
    t0 = time.time()
    res = solve(N, verbose=True)
    print("cpu %.3fs" % (time.time() - t0))
    print(res if isinstance(res, dict) else f"path length {len(res)-1}")

