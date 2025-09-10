from automaton import *
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re

from math import gcd, sqrt, log, ceil, floor 

import sys
#sys.argv = "canon.py table --cols n,-factorzig,numdiv,sumdiv,eulertot,carmichael,dedekind,mob,numfacs,totfacs,liouville,mertens,vonmangoldt,rad,m*,-psym --max 1020 --factor-exponents --factor-dots --csv --pad --column-info --header-line --residues 11 --residue-two --column-symbols".split()


import argparse
arg = argparse.ArgumentParser()
arg.add_argument("command", type=str, choices=['table', 'orbit', 'burndown', 'twins', 'columns' ], help="The command to run")
arg.add_argument("--cols", type=str, default='oc,k,dist,factors,orbit')
arg.add_argument("--min", type=int, default=1)
arg.add_argument("--max", type=int, default=100)
arg.add_argument("--odds", action="store_true")
arg.add_argument("--output-file", type=str, default=None)
arg.add_argument("--column-info", action="store_true")
arg.add_argument("--header-line", action="store_true")
arg.add_argument("--trailing-comma", action="store_true")
arg.add_argument("--pad", action="store_true")
arg.add_argument("--autocompare", action="store_true")
arg.add_argument("--factor-braces", action="store_true")
arg.add_argument("--factor-exponents", action="store_true")
arg.add_argument("--factor-dots", action="store_true")
arg.add_argument("--residues", type=int, default=0)
arg.add_argument("--residue-two", action="store_true", help="Use 2 as the first residue prime, not 3")
arg.add_argument("--odd-core-info", action="store_true")
arg.add_argument("--odd-core-info-only", action="store_true")
arg.add_argument("--no-delimiters", action="store_true")
arg.add_argument("--csv", action="store_true")
arg.add_argument("--sort-by", type=str, default=None)
arg.add_argument("--n", type=int, default=27)
arg.add_argument("--burn-span", type=int, default=1000)
arg.add_argument("--burn-lead", type=int, default=1)
arg.add_argument("--burn-base", type=int, default=2)
arg.add_argument("--burn-symbols", type=str, default='░█')
arg.add_argument("--burn-reverse", action="store_true")
arg.add_argument("--count-factors", action="store_true")
arg.add_argument("--radical-only", action="store_true")
arg.add_argument("--mask-limit", type=int, default=80)
arg.add_argument("--mask-symbols", type=str, default=' X')
arg.add_argument("--column-symbols", action="store_true")
arg.add_argument("--psym-limit", type=int, default=12)
arg = arg.parse_args()


#arg.command = 'table'
#arg.cols = 'factors'

#arg.max = 1000
#arg.n = 871
##arg.cols = 'n,base2,base3,factors,dist,nl,ocldist,nr,ocrdist,orbit_back'
##arg.cols = '-dist,-n,factors,-ocldist,-nl,ofactorsl,-ocrdist,-nr,ofactorsr,-tcn,-base2,-base3,-base5,-base7,m*,-pre_total'

# factorzig,pi,-piest,-mob,-numfacs,-totfacs,-squarepart,-liouville,-numdiv,-sumdiv,-eulertot,-carmichael,-vonmangoldt,-riemannr,-mertens,-cheby,-chebypow,-rad,m

#arg.cols='-n,pclass,factors'
##arg.trailing_comma = True
##arg.sort_by = 'factors'
##arg.odds = True
#arg.column_info = True
#arg.header_line = True
##arg.pad = True
##arg.autocompare = True
#arg.factor_braces = True
#arg.factor_exponents = True
#arg.factor_dots = True
#arg.residues = 10
##arg.pad = True
#arg.csv = True
#arg.count_factors = True
#arg.radical_only = True


from sympy import mobius, li, log, Integer, floor, primerange, nextprime, isprime
from sympy.ntheory import divisor_count, divisor_sigma, totient, factorint


def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(10000)


symbol_map = {
    2: '░█',
    3: '░▒█',
}

def padic(n, base = 2):
    k = 0
    if n:
        while n % base == 0:
            n = n // base
            k += 1
    return n, k

def odd_val(n):
    return padic(n, base=2)

def oc(n):
    oc, _ = padic(n, base=2)
    return oc

def k(n):
    _, k = padic(n, base=2)
    return k

def is_pow_two(n):
    return n & (n - 1) == 0

def collatz_classic(n):
    if n % 2:
        return 3 * n + 1
    else:
        return n // 2

def collatz_odd(n):
    return oc(3 * n + 1)

def collatz_odd_dist(n):
    dist = 0
    while n > 1:
        dist += 1
        n = collatz_odd(n)
    return dist

def collatz_orbit_list(n):
    orbit = [n]
    while n > 1:
        n = collatz_odd(n)
        orbit.append(n)
    return orbit

def collatz_orbit_str(n, sep = ' ->', count = None):
    suffix = ''
    elements = collatz_orbit_list(n)
    if count is not None and len(elements) > count:
        elements = elements[:count]
        suffix = '...'
    return sep.join(map(str, elements)) + suffix

def collatz_orbit_str_backwards(n, sep = '<- '):
    return sep.join(map(str, reversed(collatz_orbit_list(n))))

def get_superscript(digits):
    sup = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    result = ''
    for digit in str(digits):
        if digit.isdigit():
            result += sup[int(digit)]
        else:
            result += digit
    return result

def left_digits(s):
    digits = ''
    for c in s:
        if c.isdigit():
            digits += c
        else:
            break
    return digits

factorcache = {}
def _factorint(n):
    global factorcache
    if not n in factorcache:
        factorcache[n] = factorint(n)
    return factorcache[n]

def get_prime_factor_list(n):
    factorization =  _factorint(n)
    return sorted(list(factorization.keys()))

def get_prime_factors(n):
    factorization =  _factorint(n)
    result = []
    for p, exp in factorization.items():
        pstr = str(p)
        if arg.factor_exponents and exp > 1:
            pstr += get_superscript(exp)
        result.append(pstr)
    return result




def get_factor_mask(n):
    mask = ''
    for pidx, p in enumerate(PRIMES):
        if pidx > n:
            break
        if n % p == 0:
            mask += arg.mask_symbols[1]
        else:
            mask += arg.mask_symbols[0]
        if len(mask) >= arg.mask_limit:
            break
    return mask

def get_str_in_radix(n, base):
    if n == 0:
        return '0'
    digits = ''
    while n > 0:
        digits += f'{n % base}'
        n //= base
    return digits[::-1]

def get_radical(n):
    rad = 1
    for p in get_prime_factor_list(n):
        rad *= p
    return rad

def reverse_string(s):
    return s[::-1]

def align_delimeters(items, delim=','):
    col_width = {}
    for item in enumerate(items):
        elems = item.split(delim)
        for i, elem in enumerate(elems):
            if i not in col_width:
                col_width[i] = 0
            col_width[i] = max(col_width[i], len(elem))
    aligned_items = []
    for item in items:
        elems = item.split(delim)
        aligned_item = ''
        for i, elem in enumerate(elems):
            if i > 0:
                aligned_item += delim
            aligned_item += elem.rjust(col_width[i])
        aligned_items.append(aligned_item)
    return aligned_items


def get_prime_ring_cycles(p):

    fixed = (p - 1) // 2
    index = {fixed: 0}
    
    idx = 1
    n = 0
    while len(index.keys()) < p:
        if n in index:
            idx += 1
            n = 1
            while n in index:
                n += 1
            # set n to the lowest value not yet in cycle_index

        index[n] = idx
        n = (3 * n + 1) % p

    return index

prime_cycles = {}

def get_prime_ring_cycle(p, n):
    global prime_cycles
    if p not in prime_cycles:
        prime_cycles[p] = get_prime_ring_cycles(p)
    cycles = prime_cycles[p]
    return cycles[n]


def get_odd_to_odd_predecessors(n, count = 4):
    preimages = []
    if n % 3 != 0:
        while len(preimages) < count:
            n *= 2
            if (n - 1) % 3 == 0:
                preimages.append((n - 1) // 3)
    return preimages


indirect_preimages = {}
def calc_indirect_preimages(ns):
    global indirect_preimages

    indirect_preimages = {}

    for n in ns:
        i = n
        while True:
            i = collatz_odd(i)
            if i not in indirect_preimages:
                indirect_preimages[i] = 0
            indirect_preimages[i] += 1
            if i == 1:
                break

def is_primorial(n):
    if n < 2:
        return False
    
    factors = get_prime_factors(n)
    if len(factors) == 0 or factors[0] != 2:
        return False
    
    for i in range(1, len(factors)):
        if factors[i] != nextprime(factors[i - 1]):
            return False
    
    return True


def get_prime_specialness(n, verbose = True):
    if n > 2:
        factors = get_prime_factor_list(n)

        if len(factors) == 1:
            if n == factors[0]:
                if isprime(n-2) or isprime(n+2):
                    return 'prime twin' if verbose else 'pt'
                
                return 'prime' if verbose else 'p'
            else:
                return 'prime power' if verbose else 'pp'
            
        radical = get_radical(n)
        if radical == n:
            if is_primorial(n):
                return 'primorial' if verbose else 'm'
            else:
                return 'radical' if verbose else 'r'

        nr = n
        while nr % radical == 0:
            nr //= radical
            if nr == 1:
                if is_primorial(radical):
                    return 'primorial power' if verbose else 'mp'
                else:
                    return 'radical power' if verbose else 'rp'
    
    return ''
    
def get_symmetric_primes(n, limit = 5, range = 30):
    d = 1
    result = []
    while d <= range:
        if isprime(n - d) and isprime(n + d):
            result.append(d)
            if len(result) >= limit:
                break
        d += 1
    return result


def get_symmetric_prime_str(n):
    dlist = get_symmetric_primes(n, limit=arg.psym_limit + 1, range=n - 1)
    suffix = ''
    if len(dlist) > arg.psym_limit:
        dlist = dlist[:arg.psym_limit]
        unicode_ellipsis = ' \u2026'
        suffix = unicode_ellipsis
    return ', '.join(map(str, dlist)) + suffix

def get_goldbach_pairs(n):
    pairs = []
    if n % 2 == 0 and n > 2:
        for p in primerange(2, n // 2 + 1):
            q = n - p
            if isprime(q):
                pairs.append((p, q))
    return pairs

def get_goldbach_pair_count(n):
    return len(get_goldbach_pairs(n))

def get_goldbach_pair_string(n):
    pairs = get_goldbach_pairs(n)
    if pairs:
        pair_strs = [f'{q}+{p}' for p, q in pairs]
        return ' '.join(pair_strs)
    else:
        return ''

def get_goldbach_d_string(n):
    pairs = get_goldbach_pairs(n)
    if pairs:
        pair_strs = []
        for p, _ in pairs:
            d = n // 2 - p
            pair_strs.append(str(d))
        return ' '.join(reversed(pair_strs))
    else:
        return ''

def get_closest_prime(n):
    if n < 2:
        return 2
    if isprime(n):
        return n
    lower = n - 1
    while lower > 1 and not isprime(lower):
        lower -= 1
    upper = n + 1
    while not isprime(upper):
        upper += 1
    if n - lower <= upper - n:
        return lower
    else:
        return upper

def get_dist_to_lower_prime(n):
    if n < 2:
        return 2 - n
    if isprime(n):
        return 0
    lower = n - 1
    while lower > 1 and not isprime(lower):
        lower -= 1
    return n - lower




def get_dist_to_higher_prime(n):
    if n < 2:
        return 2 - n
    if isprime(n):
        return 0
    upper = n + 1
    while not isprime(upper):
        upper += 1
    return upper - n

def get_dist_to_prime(n):
    to_lower  = get_dist_to_lower_prime(n)
    to_higher = get_dist_to_higher_prime(n)
    return min(to_lower, to_higher)


def get_factor_of_coprimality(n, radicals_only = False):
    limit = int(sqrt(n))
    total = 0
    coprime = 0
    for r in range(1, limit + 1):
        if radicals_only and get_radical(r) != r:
            continue
        total += 1
        if gcd(n, r) == 1:
            coprime += 1
    if total > 0:
        return coprime / total
    else:
        return 0
    
def get_factor_of_radical_coprimality(n):
    return get_factor_of_coprimality(n, radicals_only=True)

col_desc = {}
unique_factor_sets = {}

def get_mobeius(n):
    factors =  _factorint(n)
    for exp in factors.values():
        if exp > 1:
            return 0
    return -1 if len(factors) % 2 else 1

def get_prime_counting(n):
    count = 0
    for p in primerange(1, n + 1):
        count += 1
    return count



def get_unique_factor_count(n: int) -> int:
    # ω(n): number of distinct prime factors
    return len( _factorint(n))

def get_factor_count_with_multiplicity(n: int) -> int:
    # Ω(n): total prime factors counting multiplicity
    return sum( _factorint(n).values())

def get_squarepart(n: int) -> int:
    # squarepart = n / rad(n) = ∏ p^(e-1) over p^e || n  (as you defined)
    fac =  _factorint(n)
    res = 1
    for p, e in fac.items():
        res *= p ** max(e - 1, 0)
    return res

def get_liouville(n: int) -> int:
    # Liouville λ(n) = (-1)^Ω(n)
    return -1 if get_factor_count_with_multiplicity(n) % 2 else 1

def get_num_divisors(n: int) -> int:
    # τ(n)
    return int(divisor_count(n))

def get_sum_of_divisors(n: int) -> int:
    # σ(n)
    return int(divisor_sigma(n))

def get_euler_totient(n: int) -> int:
    # φ(n)
    return int(totient(n))

def get_carmichael(n: int) -> int:
    # Carmichael λ(n) via lcm of prime-power components
    from math import lcm
    fac =  _factorint(n)
    parts = []
    for p, e in fac.items():
        if p == 2:
            if e == 1:
                parts.append(1)
            elif e == 2:
                parts.append(2)
            else:
                parts.append(2 ** (e - 2))
        else:
            parts.append((p - 1) * (p ** (e - 1)))
    if not parts:
        return 1
    out = parts[0]
    for v in parts[1:]:
        out = lcm(out, v)
    return out

def get_von_mangoldt(n: int):
    # Λ(n) = log p if n is a prime power p^k, else 0
    if n < 2:
        return 0
    fac =  _factorint(n)
    if len(fac) == 1:
        (p, e), = fac.items()
        if e >= 1:
            return log(p)
    return 0

def get_riemann_li(n: int):
    # logarithmic integral Li(n), principal value
    if n < 2:
        return 0
    return li(n)

def get_riemann_r(n: int):
    # Riemann R function via series: R(x) = Σ_{k≥1} μ(k)/k · li(x^(1/k))
    # truncate at k where x^(1/k) < 2  ⇒ k ≤ floor(log_2 x)
    if n <= 0:
        return 0
    if n < 2:
        return 0
    K = int(floor(log(int(n), 2)))
    total = 0
    for k in range(1, K + 1):
        mu = mobius(k)
        if mu != 0:
            total += (mu / k) * li(n ** (1 / k))
    return total

def get_mertens(n: int) -> int:
    # M(n) = Σ_{k≤n} μ(k)
    if n < 1:
        raise ValueError("n must be a positive integer")
    s = 0
    for k in range(1, n + 1):
        s += mobius(k)
    return int(s)

def get_chebyshev_primes(n: int):
    # θ(n) = Σ_{p≤n} log p
    if n < 2:
        return 0
    total = 0
    for p in primerange(2, n + 1):
        total += log(p)
    return total

def get_chebyshev_prime_powers(n: int):
    # ψ(n) = Σ_{p^k ≤ n} log p = Σ_{p≤n} floor(log_p n) · log p
    if n < 2:
        return 0
    total = 0
    for p in primerange(2, n + 1):
        kmax = floor(log(n, p))
        if kmax > 0:
            total += kmax * log(p)

    return total


from math import log, floor, sqrt
from sympy import primerange

def get_chebyshev_prime_powers_fast(n: int, rel_tol: float = 1e-3) -> float:
    """
    Fast approximation of Chebyshev's psi function:
        ψ(n) = sum_{p^k ≤ n} log p = sum_{p ≤ n} floor(log_p n) * log p

    Strategy:
      1) For primes p ≤ √n (where k_max ≥ 2), compute contributions exactly.
      2) For the big 'k=1' tail (p > √n), approximate θ(x) = sum_{p ≤ x} log p by θ(x) ≈ x.
         This turns the heavy tail into (n - θ(√n)) ≈ (n - θ_exact(√n)).
      3) To secure ~3-decimal relative accuracy, correct the last rel_tol·n window exactly,
         replacing its approximation with the exact sum of log p there.

    rel_tol=1e-3 targets ~0.1% relative error (≈ 3 decimals on ψ(n) ~ n).
    Returns a float.
    """
    if n < 2:
        return 0.0

    # 1) Exact contributions from small primes (p ≤ √n),
    #    where k_max = floor(log(n, p)) ≥ 2
    y = int(sqrt(n))
    small_sum = 0.0
    theta_y = 0.0  # exact θ(y) while we're at it

    for p in primerange(2, y + 1):
        lp = log(p)
        kmax = int(floor(log(n, p)))  # number of prime-power hits for this p
        small_sum += kmax * lp
        theta_y += lp

    # 2) Approximate the large 'k=1' tail by θ(n) - θ(y) ≈ n - θ(y)
    psi = small_sum + (n - theta_y)

    # 3) Replace the *last* rel_tol·n of that tail by its exact value
    #    (this keeps the overall error within ~rel_tol · n)
    m = int(n * (1.0 - rel_tol))
    if m > y:
        # remove the approximated chunk length (n - m) we added implicitly,
        # then add back its exact value: sum_{m < p ≤ n} log p
        approx_last_window = (n - m)
        tail_exact = 0.0
        for p in primerange(m + 1, n + 1):
            tail_exact += log(p)
        psi = small_sum + (n - theta_y - approx_last_window) + tail_exact

    return float(psi)


def get_dedekind_psi(n: int):
    # Dedekind ψ(n) = n · ∏_{p|n} (1 + 1/p)
    if n < 1:
        raise ValueError("n must be a positive integer")
    fac =  _factorint(n)
    prod = 1
    for p in fac.keys():
        prod *= (1 + 1 / p)
    return int(n * prod)

def get_largest_prime_factor(n):
    if n < 2:
        return None
    fac =  _factorint(n)
    return max(fac.keys())

def get_smallest_prime_factor(n):
    if n < 2:
        return None
    fac =  _factorint(n)
    return min(fac.keys())


def get_residue_volatility_abs(n):
    volatility = 0
    for p in primerange(2, int(sqrt(n)) + 1):
        volatility += n % p
    return volatility


def get_residue_volatility(n):
    volatility = 0
    for p in primerange(2, int(sqrt(n)) + 1):
        r = n % p
        volatility += r * 1.0 / p
    return volatility

def get_residue_volatility_signed(n):
    volatility = 0
    for p in primerange(2, int(sqrt(n)) + 1):
        r = n % p
        if r > p // 2:
            r -= p
        volatility += r * 1.0 / p
    return volatility


def get_harmonic_sum(n):
    # sum 1/p for all factors p
    if n < 2:
        return 0.0
    fac =  _factorint(n)
    total = 0.0
    for p in fac.keys():
        total += 1.0 / p
    return total

def get_harmonic_sum_inv(n):
    s = get_harmonic_sum(n)
    if s != 0:
        return 1.0 / s
    return 0.0

def estimate_primes_under_n(n):
    if n < 2:
        return 0
    return n / (log(n) - 1)

def radix_name(radix):
    if radix == 2:
        return 'binary'
    elif radix == 3:
        return 'ternary'
    else:
        return f'base-{radix}'

def calc_n_values(n):
    global col_desc
    val = {}

    col_desc['n'] = f'a positive{" odd" if arg.odds else ""} integer'
    val['n'] = n

    oc, k = odd_val(n)

    col_desc['oc'] = f'the "odd core" of positive integer n, after dividing out powers of two'
    val['oc'] = oc

    col_desc['k'] = f'the 2-adic valuation of n, such that n = oc * 2^k'
    val['k'] = k

    col_desc['v2'] = f'the 2-adic valuation of n'
    val['v2'] = k

    col_desc['tc'] = f'the "ternary core" of n, after dividing out powers of three'
    tc3, v3 = padic(n, base=3)
    val['tc'] = tc3

    col_desc['tcn'] = f'the nearest "ternary core", among {n-1, n, n+1}, after dividing out powers of three'
    tcn = (n - 1) if (n - 1) % 3 == 0 else n if n % 3 == 0 else (n + 1)
    tcn3, vn3 = padic(tcn, base=3)
    val['tcn'] = tcn3


    col_desc['v3'] = f'the 3-adic valuation of n, such that n = tc * 3^v3'
    val['v3'] = v3

    if arg.odd_core_info:
        paramname = 'oc(n)'
        param = oc
    else:
        paramname = 'n'
        param = n

    col_desc['pre'] = f'the preimages of {paramname} under the odd-to-odd Collatz map'
    preimages = get_odd_to_odd_predecessors(param)
    val['pre'] = '{' + (', '.join([str(i) for i in preimages]) + '...' if preimages else '') + '}'

    col_desc['pre_total'] = f'the total number of preimages of {paramname} under the odd-to-odd Collatz map'
    val['pre_total'] = indirect_preimages.get(param, 0)

    col_desc['dist'] = f'the number of odd-to-odd Collatz for {paramname} steps to reach 1'
    val['dist'] = collatz_odd_dist(param)

    col_desc['next'] = f'the next odd-to-odd Collatz step of {paramname}'
    val['next'] = collatz_odd_next(param)

    col_desc['orbit'] = f'the remaining orbit of {paramname} under the odd-to-odd Collatz map, all the way to 1'
    val['orbit'] = collatz_orbit_str(param)

    col_desc['orbit_back'] = f'the remaining orbit of {paramname} under the odd-to-odd Collatz map, backwards from 1'
    val['orbit_back'] = collatz_orbit_str_backwards(param)

    col_desc['orbitpeek'] = f'the next steps in the orbit of {paramname} under the odd-to-odd Collatz map'
    val['orbitpeek'] = collatz_orbit_str(param, count=7)

    col_desc['primeness'] = f'notes if {paramname} is a prime, prime power, prime twin, radical, radical power, primorial, or primorial power'
    val['primeness'] = get_prime_specialness(param)

    col_desc['pclass'] = f'notes if {paramname} is a prime (p), prime power (pp), prime twin (pt), radical (r), radical power (rp), primorial (m), or primorial power (mp)'
    val['pclass'] = get_prime_specialness(param, verbose=False)

    col_desc['psym'] = f'the symmetric primes at equal offsets around {paramname}'
    val['psym'] = get_symmetric_prime_str(param)

    col_desc['psymext'] = f'the symmetric primes p=n-d and q=n+d in the immediate vicinity of {paramname}, given as p:q(d)'
    psymnext = ''
    psym = get_symmetric_primes(param, limit=arg.psym_limit + 1, range=100)
    for d in psym:
        psymnext += f'{param - d}:{param + d}({d}) '
    val['psymext'] = psymnext.strip()

    col_desc['gbpairs'] = f'the Goldbach prime pairs that sum to {paramname}, if {paramname} is even and greater than 4'
    val['gbpairs'] = get_goldbach_pair_string(param) if param % 2 == 0 and param > 4 else ''

    col_desc['gbcount'] = f'the number of Goldbach prime pairs that sum to {paramname}, if {paramname} is even and greater than 4'
    val['gbcount'] = get_goldbach_pair_count(param) if param % 2 == 0 and param > 4 else 0

    col_desc['gbds'] = f'the offsets of prime pairs around {paramname}/2, which represent Goldbach partitions for n, if {paramname} is even and greater than 4'
    val['gbds'] = get_goldbach_d_string(param) if param % 2 == 0 and param > 4 else ''

    col_desc['cop'] = f'the factor of coprimality of {paramname} with all integers up to sqrt({paramname})'
    val['cop'] = f'{get_factor_of_coprimality(param):.3f}'

    col_desc['rcop'] = f'the factor of coprimality of {paramname} with all radicals up to sqrt({paramname})'
    val['rcop'] = f'{get_factor_of_radical_coprimality(param):.3f}'

    col_desc['mob'] = f'the Möbius function of {paramname}, which is 1 if {paramname} is a product of an even number of distinct primes, -1 if odd, and 0 if any prime is repeated'
    val['mob'] = get_mobeius(param)
    
    col_desc['pi'] = f'the prime counting function π({paramname}), the number of primes less than or equal to {paramname}'
    val['pi'] = get_prime_counting(param)

    col_desc['piest'] = f'an estimate of the prime counting function π({paramname}), the number of primes less than or equal to {paramname}: n / (log n - 1)'
    piest = estimate_primes_under_n(param)
    val['piest'] = f'{piest:.3f}'

    col_desc['piesti'] = f'an estimate n / (log n - 1) of the prime counting function π({paramname}),the number of primes less than or equal to {paramname}, truncated to an integer'
    val['piesti'] = int(piest)

    col_desc['pierr'] = f'the error in the estimate of the prime counting function π({paramname})'
    pierr = val['pi'] - piest
    val['pierr'] = f'{pierr:.3f}'

    col_desc['primedist'] = f'the distance from {paramname} to the nearest prime'
    val['primedist'] = get_dist_to_prime(param)

    col_desc['numfacs'] = f'the number of distinct prime factors of {paramname}, ω({paramname})'
    val['numfacs'] = get_unique_factor_count(param)

    col_desc['totfacs'] = f'the total number of prime factors of {paramname}, counting multiplicity, Ω({paramname})'
    val['totfacs'] = get_factor_count_with_multiplicity(param)

    col_desc['squarepart'] = f'the squarepart of {paramname}, the largest perfect square dividing {paramname}'
    val['squarepart'] = get_squarepart(param)

    col_desc['liouville'] = f'the Liouville function of {paramname}, λ({paramname}) = (-1)^Ω({paramname})'
    val['liouville'] = get_liouville(param)

    col_desc['numdiv'] = f'the number of divisors of {paramname}, τ({paramname})'
    val['numdiv'] = get_num_divisors(param)

    col_desc['sumdiv'] = f'the sum of divisors of {paramname}, σ({paramname})'
    val['sumdiv'] = get_sum_of_divisors(param)

    col_desc['eulertot'] = f'the Euler totient of {paramname}, φ({paramname})'
    val['eulertot'] = get_euler_totient(param)

    col_desc['carmichael'] = f'the Carmichael function of {paramname}, λ({paramname})'
    val['carmichael'] = get_carmichael(param)

    col_desc['vonmangoldt'] = f'the von Mangoldt function of {paramname}, Λ({paramname})'
    val['vonmangoldt'] = f'{get_von_mangoldt(param):.3f}'

    col_desc['riemannli'] = f'the logarithmic integral Li({paramname})'
    val['riemannli'] = f'{get_riemann_li(param):.3f}'

    col_desc['riemannr'] = f'the Riemann R function R({paramname})'
    val['riemannr'] = f'{get_riemann_r(param):.3f}'

    col_desc['mertens'] = f'the Mertens function M({paramname})'
    val['mertens'] = get_mertens(param)

    col_desc['cheby'] = f'the Chebyshev θ({paramname}) function, the sum of log p for primes p ≤ {paramname}'
    val['cheby'] = f'{get_chebyshev_primes(param):.3f}'

    col_desc['chebypow'] = f'the Chebyshev ψ({paramname}) function, the sum of log p for prime powers p^k ≤ {paramname}'
    val['chebypow'] = f'{get_chebyshev_prime_powers_fast(param):.3f}'

    col_desc['dedekind'] = f'the Dedekind ψ({paramname}) function'
    val['dedekind'] = get_dedekind_psi(param)

    col_desc['lpf'] = f'the largest prime factor of {paramname}'
    val['lpf'] = get_largest_prime_factor(param) or ''

    col_desc['spf'] = f'the smallest prime factor of {paramname}'
    val['spf'] = get_smallest_prime_factor(param) or ''

    col_desc['rvol'] = f'the residue volatility of {paramname}, the sum of the prime residues of {paramname} weighted by 1/p'
    val['rvol'] = f'{get_residue_volatility(param):.3f}'

    col_desc['rvolabs'] = f'the absolute residue volatility of {paramname}, the sum of the prime residues of {paramname}'
    val['rvolabs'] = f'{get_residue_volatility_abs(param):.3f}'

    col_desc['rvols'] = f'the signed residue volatility of {paramname}, the sum of the signed prime residues of {paramname} weighted by 1/p'
    val['rvols'] = f'{get_residue_volatility_signed(param):.3f}'

    col_desc['hsum'] = f'the harmonic sum, the sum of the reciprocals of the prime factors of {paramname}'
    val['hsum'] = f'{get_harmonic_sum(param):.5f}'

    col_desc['hsuminv'] = f'inverse of the harmonic sum of {paramname} (the sum of the reciprocals of the prime factors)'
    val['hsuminv'] = f'{get_harmonic_sum_inv(param):.3f}'

    col_desc['m*'] = f'the prime residues of {paramname} modulo a set of primes'
    col_desc['sm*'] = f'the signed prime residues of {paramname} modulo a set of primes'
    col_desc['cyc*'] = f'the prime ring cycles of {paramname}'
    residue_offset = 0 if arg.residue_two else 1
    if arg.residues > 0:
        for r in range(arg.residues):
            p = PRIMES[r + residue_offset]
            val['m' + str(p)] = param % p
            val['sm' + str(p)] = (param % p) - p if (param % p) > p // 2 else (param % p)
            val['cyc' + str(p)] = get_prime_ring_cycle(p, param % p)

    for radix in [2, 3, 5, 7, 11]:
        col_desc['base' + str(radix)] = f'the {radix_name(radix)} representation of {paramname}'
        in_radix = get_str_in_radix(param, radix)
        val['base' + str(radix)] = in_radix

        col_desc['base' + str(radix) + 'rev'] = f'the {radix_name(radix)} representation of {paramname}, reversed'
        val['base' + str(radix) + 'rev'] = reverse_string(in_radix)


    for radix in [2, 3]:
        symbol = get_str_in_radix(param, radix)
        remap = symbol_map[radix]
        for i, char in enumerate(remap):
            symbol = symbol.replace(str(i), char)

        delimiter_desc = ''
        symbol_name = 'symbol' + str(radix)
        if radix == 2:
            if arg.no_delimiters:
                symbol = symbol[1:-1]
                delimiter_desc = ' (without MSB and LSB)'

        key_desc = ''
        for i in range(len(remap)):
            idx = i + 1
            if key_desc:
                key_desc += ', '
            key_desc += f'{idx} = "{remap[i]}"'

        col_desc[symbol_name] = f'symbolic representation{delimiter_desc} of the {radix_name(radix)} encoding of {paramname} ({key_desc})'
        val[symbol_name] = symbol

    all_factors = get_prime_factors(param)

    facsep = ', '
    if arg.factor_dots:
        facsep = '⋅'

    factors_str = facsep.join([str(f) for f in all_factors])
    if factors_str not in unique_factor_sets:
        unique_factor_sets[factors_str] = 0
    unique_factor_sets[factors_str] += 1

    cursor_factors = get_prime_factors(param)
    for exclude in [2, 3]:
        if exclude in cursor_factors:
            cursor_factors.remove(exclude)

    cfactors_str =  facsep.join([str(f) for f in cursor_factors])
    factorsl_str =  facsep.join([str(f) for f in get_prime_factors(param - 1)])
    factorsr_str =  facsep.join([str(f) for f in get_prime_factors(param + 1)])
    ofactorsl_str = facsep.join([str(f) for f in get_prime_factors(param - 1) if str(f)[0] != '2'])
    ofactorsr_str = facsep.join([str(f) for f in get_prime_factors(param + 1) if str(f)[0] != '2'])

    if arg.factor_braces:
        factors_str = '{' + factors_str + '}'
        cfactors_str = '{' + cfactors_str + '}'
        factorsl_str = '{' + factorsl_str + '}'
        factorsr_str = '{' + factorsr_str + '}'
        ofactorsl_str = '{' + ofactorsl_str + '}'
        ofactorsr_str = '{' + ofactorsr_str + '}'

    col_desc['factors'] = f'the prime factors of {paramname}'
    val['factors'] = factors_str


    col_desc['facgbds'] = f'factorizations of the offsets of prime pairs around {paramname}/2, which represent Goldbach partitions for n, if {paramname} is even and greater than 4'

    # facgdbs is a list of integers separated by spaces. Replace each integer with its factorization, using facsep
    facgdbs_list = val['gbds'].split(' ')
    facgdbs_factors = []
    if facgdbs_list and facgdbs_list[0] != '':
        for dstr in facgdbs_list:
            d = int(dstr)
            dfactors = get_prime_factors(d)
            dfactors_str = facsep.join([str(f) for f in dfactors])
            if arg.factor_braces:
                dfactors_str = '{' + dfactors_str + '}'
            facgdbs_factors.append(dfactors_str)
    val['facgbds'] = ' '.join(facgdbs_factors)

    col_desc['closestprime'] = f'the closest prime to {paramname}'
    closest = get_closest_prime(param)
    val['closestprime'] = closest

    col_desc['factorzig'] = f'the prime factors of {paramname}, indented by distance from the nearest prime'
    dist = abs(param - closest)
    val['factorzig'] = ' ' * dist + factors_str

    col_desc['factorsl'] = f'the prime factors of {paramname}-1'
    val['factorsl'] = factorsl_str

    col_desc['ofactorsl'] = f'the odd prime factors of {paramname}-1'
    val['ofactorsl'] = ofactorsl_str

    col_desc['factorsr'] = f'the prime factors of {paramname}+1'
    val['factorsr'] = factorsr_str

    col_desc['ofactorsr'] = f'the odd prime factors of {paramname}+1'
    val['ofactorsr'] = ofactorsr_str

    col_desc['cfactors'] = f'the prime factors above 3 of {paramname}'
    val['cfactors'] = cfactors_str

    col_desc['rad'] = f'the radical of {paramname}, the product of its unique prime factors'
    rad = 1
    for p in get_prime_factor_list(param):
        rad *= p
    val['rad'] = rad

    col_desc['crad'] = f'the Collatz "cursor radical" of {paramname}, the product of its prime factors above 3'
    crad = 1
    for p in get_prime_factor_list(param):
        if p > 3:
            crad *= p   
    val['crad'] = crad

    col_desc['mask'] = f'the mask of prime factors (radical) of {paramname}, as a string'
    mask = get_factor_mask(param)
    val['mask'] = mask

    col_desc['cmask'] = f'mask of Collatz "cursor radical" of {paramname}, as a string'
    cmask = mask[2::]
    val['cmask'] = cmask

    t = create_automaton(n)
    ocl, kl, ocr, kr = t
    compare = ' > ' if kl > kr else ' < '

    col_desc['mech'] = f'the Collatz automaton of n, as a tuple of four integers'
    val['mech'] = f'{ocl} ({kl}{compare if arg.autocompare else " "}{kr}) {ocr}'

    col_desc['mech_next'] = f'the Collatz automaton prediction of the next step'
    tn = step_automaton(t)
    ocln, kln, ocrn, krn = tn
    val['mech_next'] = f'{ocln} ({kln}{compare if arg.autocompare else " "}{krn}) {ocrn}'


    col_desc['fingerprint'] = f'the fingerprint of the Collatz automaton of n, as a pair of 2-adic values'
    val['fingerprint'] = f'({ocl}, {kl}) ({ocr}, {kr})'

    col_desc['ocl'] = f'the odd core of n-1 (the even number to the left of odd n)'
    val['ocl'] = ocl
    col_desc['kl'] = f'the k value n-1 (the even number to the left of odd n)'
    val['kl'] = kl
    col_desc['ocr'] = f'the odd core of n+1 (the even number to the right of odd n)'
    val['ocr'] = ocr
    col_desc['kr'] = f'the k value n+1 (the even number to the right of odd n)'
    val['kr'] = kr

    oclo, klo = (ocl, kl) if kl < kr else (ocr, kr)
    ochi, khi = (ocr, kr) if kl < kr else (ocl, kl)

    col_desc['oclo'] = f'the odd core of the even number n-1 to the left of odd n, with lowest k value'
    val['oclo'] = oclo
    col_desc['ochi'] = f'the odd core of the even number n+1 to the right of odd n, with highest k value'
    val['ochi'] = ochi
    col_desc['klo'] = f'the k value of the odd core of the even number n-1 to the left of odd n, with lowest k value'
    val['klo'] = klo
    col_desc['khi'] = f'the k value of the odd core of the even number n+1 to the right of odd n, with highest k value'
    val['khi'] = khi

    col_desc['nl'] = f'the odd core valuation of the even number n-1 to the left of odd n'
    val['nl'] = f'({ocl}, {kl})'
    col_desc['nr'] = f'the odd core valuation of the even number n+1 to the right of odd n'
    val['nr'] = f'({ocr}, {kr})'

    col_desc['ocldist'] = f'the Collatz odd-to-odd distance of the odd core of n-1'
    val['ocldist'] = collatz_odd_dist(ocl)
    col_desc['ocrdist'] = f'the Collatz odd-to-odd distance of the odd core of n+1'
    val['ocrdist'] = collatz_odd_dist(ocr)

    col_desc['mech_sorted'] = col_desc['mech'] + ', lowest k value first'
    val['mech_sorted'] = f'{oclo} ({klo}{" < " if arg.autocompare else " "}{khi}) {ochi}'

    col_desc['print_sorted'] = col_desc['fingerprint'] + ', lowest k value first'
    val['print_sorted'] = f'({oclo}, {klo}) ({ochi}, {khi})'

    return val

n_cache = {}
def get_n_values(n):
    global n_cache
    if not n in n_cache:
        n_cache[n] = calc_n_values(n)
    return n_cache[n]

column_to_symbol = {
    'cheby':        'θ',    
    'chebypow':     'ψ',
    'dedekind':     'Dψ',
    'eulertot':     'φ',
    'pi':           'π', 
    'piest':        'π~',    
    'riemannr':     'R',    
    'riemannli':    'Li',   
    'mob':          'μ',  
    'numfacs':      'ω',    
    'totfacs':      'Ω',    
    'liouville':    'λ',    
    'mertens':      'M',    
    'numdiv':       'τ',    
    'sumdiv':       'σ',    
    'carmichael':   'λ',    
    'vonmangoldt':  'Λ',    
    'lpf':          'P+',
    'spf':          'P−',
}

ljust_by_default = [
    'factor'
    'psym',
    'mask',
]


_SUPERS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUP_TO_NORMAL = str.maketrans(_SUPERS, "0123456789")
_PAIR_RE = re.compile(r"([0-9]+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*)")

def parse_factorization(s: str):
    # "{2⁴ 3 5²³}" → [2, 4, 3, 1, 5, 23]
    terms = []
    for base, sup in _PAIR_RE.findall(s):
        terms.append(int(base))
        terms.append(int(sup.translate(_SUP_TO_NORMAL)) if sup else 1)
    return terms

def factorization_sort_key(s: str):
    return parse_factorization(s)




def print_output(output_lines):
    if arg.output_file:
        with open(arg.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
    else:
        for line in output_lines:
            print(line)



def print_ns(ns):
    output_lines = []
    calc_n_values(1) # populate col_desc

    calc_indirect_preimages(ns)

    sep = ','

    cols = arg.cols.split(',')
    col_info = {}
    col_headers = []
    for col in cols:
        # remove leading '-'
        col_raw = col.lstrip('-')
        col_info[col_raw] = col_desc[col_raw]

        if col_raw == 'm*' or col_raw == 'sm*' or col_raw == 'cyc*':
            if arg.residues > 0:
                residue_offset = 0 if arg.residue_two else 1
                prefix = col_raw[:-1]
                for r in range(arg.residues):
                    col_headers.append(prefix + str(PRIMES[r + residue_offset]))
        else:
            header = col_raw
            col_headers.append(header)

    header_line = ''
    cells = {}
    for col in col_headers:
        cells[col] = []
        #if arg.header_line:
        #    cells[col].append(col + sep)

    for n in ns:
        n_vals = get_n_values(n)
        for col in col_headers:
            col_val = n_vals[col]
            valstr = str(col_val)
            if arg.csv:
                valstr = valstr.replace(',', '')
            if arg.csv:
                valstr = valstr + sep
            cells[col].append(valstr)

    if arg.pad:
        for col in col_headers:
            col_width = max(len(cell) for cell in cells[col])
            col_tag = col
            if arg.column_symbols and col in column_to_symbol:
                col_tag = column_to_symbol[col]
            col_width = max(col_width, len(col_tag))
            col_width += len(sep)
            justified = []

            ljust = False
            for sub in ljust_by_default:
                if sub in col:
                    ljust = True
            if ('-' + col) in arg.cols:
                ljust = not ljust
            for cellval in cells[col]:
                if ljust:
                    justified.append(cellval.ljust(col_width))
                else:
                    justified.append(cellval.rjust(col_width))
            cells[col] = justified
            if arg.header_line:
                if arg.pad and header_line:
                    header_line += ' '
                
                if ljust:
                    header_line += (col_tag + sep).ljust(col_width)
                else:
                    header_line += (col_tag + sep).ljust(col_width)
                


    if arg.column_info:
        longest_name = max(len(name) for name in col_info.keys())
        for name, desc in col_info.items():
            label = f'`{name}`'
            output_lines.append(f'# {label.ljust(longest_name + 2)} {desc}')
        output_lines.append('')

    sorted_order = []

    if arg.sort_by in cells.keys():
        sortlist = []
        for idx, item in enumerate(cells[arg.sort_by]):
            sortlist.append((item, idx))

        if 'factors' in arg.sort_by:
            sortlist.sort(key=lambda x: factorization_sort_key(x[0]))
        else:
            sortlist.sort(key=lambda x: x[0])

        sorted_order = []
        for item, idx in sortlist:
            sorted_order.append(idx)
    else:
        for n in range(len(ns)):
            sorted_order.append(n)

    if arg.header_line:
        output_lines.append(header_line.rstrip(' ' + sep))

#    for nn in range(len(ns)):
#        n = ns[sorted_order[nn]]
    for n in sorted_order:
        line = ''
        for col in col_headers:
            line += cells[col][n]
            if arg.pad:
                line += ' '
        line = line.rstrip()
        if len(line) > 0 and line[-1] == sep.strip():
            if not arg.trailing_comma:
                line = line[:-1]
        line = line.rstrip()
        output_lines.append(line)

    print_output(output_lines)


def print_table():
    ns = []
    for n in range(arg.min, arg.max + 1):
        if arg.odds and n % 2 == 0:
            continue
        ns.append(n)
    
    print_ns(ns)

def print_orbit():
    ns = collatz_orbit_list(arg.n)
    print_ns(ns)

def print_twins():
    ns = []
    for i in range(len(PRIMES) - 1):
        if PRIMES[i] + 2 == PRIMES[i + 1]:
            val = PRIMES[i] + 1
            if arg.radical_only:
                if val != get_radical(val):
                    continue
            ns.append(val)
    print_ns(ns)

    if arg.count_factors:
        for factors_str in sorted(unique_factor_sets.keys()):
            count = unique_factor_sets[factors_str]
            print(f'{factors_str}: {count}')


def print_burndown():
    output_lines = []
    n = (1<<arg.burn_span) +((1 << arg.burn_lead) - 1)
    for i in range(1000):
        encoded = get_str_in_radix(n, arg.burn_base)
        for i, char in enumerate(arg.burn_symbols):
            encoded = encoded.replace(str(i), char)

        if arg.burn_reverse:
            backwards = ''
            for i in range(len(encoded)):
                backwards += encoded[len(encoded) - i - 1]
            output_lines.append(backwards)
        else:
            output_lines.append(encoded)
        #print(binary_n)#.rjust(200))
        n = collatz_odd(n)
        if n == 1:
            break
    print_output(output_lines)


def print_all_column_names():
    calc_n_values(1) # populate col_desc
    longest_name = max(len(name) for name in col_desc.keys())
    output_lines = []
    for name, desc in col_desc.items():
        label = f'`{name}`'
        output_lines.append(f'# {label.ljust(longest_name + 2)} {desc}')
    print_output(output_lines)

if __name__ == "__main__":
    if arg.command == "table":
        print_table()
    elif arg.command == "orbit":
        print_orbit()
    elif arg.command == "burndown":
        print_burndown()
    elif arg.command == "twins":
        print_twins()
    elif arg.command == "columns":
        print_all_column_names()


