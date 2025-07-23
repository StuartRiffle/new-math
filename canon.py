import argparse
arg = argparse.ArgumentParser()
arg.add_argument("--min", type=int, default=1)
arg.add_argument("--max", type=int, default=100)
arg.add_argument("--odds", action="store_true")
arg = arg.parse_args()

def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(1000)
ODD_PRIMES = PRIMES[1:]


symbol_map = {
    2: '-O',
    3: '-|O',
}

mask_map = '-X'

def is_pow_two(n):
    return n & (n - 1) == 0

def padic_val(base, n):
    k = 0
    if n != 0:
        while n % base == 0:
            n = n // base
            k += 1
    return n, k

def odd_core(n):
    return padic_val(2, n)

def oc(n):
    n, _ = padic_val(2, n)
    return n

def k(n):
    _, k = padic_val(2, n)
    return k

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

def collatz_orbit_list(n):
    orbit = [n]
    while n > 1:
        n = collatz_odd_next(n)
        orbit.append(n)
    return orbit

def collatz_orbit_str(n, sep = '→'):
    return sep.join(map(str, collatz_orbit_list(n)))

def get_prime_factors(n):
    factors = []
    for p in PRIMES:
        if n % p == 0:
            factors.append(p)
            n = n // p
    return factors


def get_factor_mask(n):
    mask = ''
    for pidx, p in enumerate(PRIMES):
        if pidx > n:
            break
        if n % p == 0:
            mask += mask_map[1]
        else:
            mask += mask_map[0]
    return mask

def get_str_in_radix(n, base):
    if n == 0:
        return '0'
    digits = ''
    while n > 0:
        digits += f'{n % base}'
        n //= base
    return digits[::-1]

def booktable():
    cols = 5
    rows = 100
    pages = 1


def dataset():
    pass

    # --min <num>=1 --range <num>=100 --odds 


def chart():
    rows = []
    for n in range(arg.min, arg.max + 1):
        if arg.odds and n % 2 == 0:
            continue

def get_n_values(n):
    oc, k = padic_val(2, n)
    tc3, v3 = padic_val(3, n)

    val = {}
    val['n'] = n
    val['dist'] = collatz_odd_dist(oc)
    val['oc'] = oc
    val['core2'] = oc
    val['k'] = k(n)
    val['v2'] = k
    val['tc'] = tc3
    val['tc3'] = tc3
    val['tcore'] = tc3
    val['core3'] = tc3
    val['v3'] = v3

    if arg.residues > 0:
        val['residues'] = []
        for r in range(arg.residues):
            p = PRIMES[r]
            val['m' + str(p)] = oc % p

    for radix in [2, 3, 5, 7, 11]:
        val['base' + str(radix)] = get_str_in_radix(oc, radix)

    for radix in [2, 3]:
        symbol = get_str_in_radix(oc, 2)
        if radix == 2:
            symbol = symbol[1:-1]
        remap = symbol_map[radix]
        for i, char in enumerate(remap):
            symbol = symbol.replace(char, remap[i])
        if radix == 2:
            val['symbol'] = symbol
        val['symbol' + str(radix)] = symbol

    all_factors = get_prime_factors(oc)
    factors = ' '.join(all_factors)

    cursor_factors = get_prime_factors(oc)
    cursor_factors.remove(2)
    cursor_factors.remove(3)
    cursor_factors = ' '.join(cursor_factors)

    if arg.factor_braces:
        factors = '{' + factors + '}'
        cursor_factors = '{' + cursor_factors + '}'

    val['factors'] = factors
    val['cfac'] = cursor_factors
    val['cursor'] = cursor_factors
    val['cursor_factors'] = cursor_factors

    rad = 1
    for p in all_factors:
        rad *= p
    val['rad'] = rad
    val['radical'] = rad

    crad = 1
    for p in cursor_factors:
        crad *= p
    val['crad'] = crad
    val['cradical'] = crad
    val['cursor_radical'] = crad

    mask = get_factor_mask(oc)
    val['mask'] = mask
    val['factor_mask'] = mask

    cmask = mask[2::]
    val['cmask'] = cmask
    val['cursor_mask'] = cmask

    return val




if __name__ == "__main__":

    # commands
    # canon.py booktable
    # canon.py dataset

    if arg.command == "booktable":
        booktable()
    elif arg.command == "dataset":
        dataset()


    main()
