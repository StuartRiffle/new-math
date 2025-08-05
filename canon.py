from automaton import *
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
arg = argparse.ArgumentParser()
arg.add_argument("--cols", type=str, default='oc,k,dist,factors')
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
arg.add_argument("--residues", type=int, default=0)
arg.add_argument("--odd-core-info", action="store_true")
arg.add_argument("--odd-core-info-only", action="store_true")
arg.add_argument("--no-delimiters", action="store_true")
arg.add_argument("--csv", action="store_true")
arg.add_argument("--sort-by", type=str, default=None)
arg.add_argument("--n", type=int, default=27)
arg.add_argument("--burn-span", type=int, default=200)
arg.add_argument("--burn-lead", type=int, default=1)
arg.add_argument("--burn-base", type=int, default=2)
arg.add_argument("--burn-symbols", type=str, default='░█')
arg.add_argument("--burn-reverse", action="store_true")
arg = arg.parse_args()


arg.command = 'burndown'

arg.max = 1000
arg.n = 33
#arg.cols = 'n,base2,base3,factors,dist,nl,ocldist,nr,ocrdist,orbit_back'
arg.cols = 'n,base2,base3,mech,mech_next'
#arg.sort_by = 'orbit_back'
arg.odds = True
arg.column_info = True
arg.header_line = True
#arg.pad = True
#arg.autocompare = True
arg.factor_braces = True
arg.residues = 6
arg.pad = True
arg.csv = True

def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(10000)
ODD_PRIMES = PRIMES[1:]


symbol_map = {
    2: '-O',
    3: '-|O',
}

mask_map = '-X'

def padic(n, base = 2):
    k = 0
    if n:
        while n % base == 0:
            n = n // base
            k += 1
    return n, k

def odd_core(n):
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

def collatz_orbit_str(n, sep = ' ->'):
    return sep.join(map(str, collatz_orbit_list(n)))

def collatz_orbit_str_backwards(n, sep = '<- '):
    return sep.join(map(str, reversed(collatz_orbit_list(n))))

def get_prime_factors(n):
    if n == 1367:
        print('')

    factors = set()
    for p in PRIMES:
        while n % p == 0:
            factors.add(p)
            n = n // p
        if p > n:
            break
    return sorted(list(factors))

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


col_desc = {}


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

    oc, k = odd_core(n)

    col_desc['oc'] = f'the "odd core" of positive integer n, after dividing out powers of two'
    val['oc'] = oc

    col_desc['k'] = f'the 2-adic valuation of n, such that n = oc * 2^k'
    val['k'] = k

    col_desc['v2'] = f'the 2-adic valuation of n'
    val['v2'] = k

    col_desc['tc'] = f'the "ternary core" of n, after dividing out powers of three'
    tc3, v3 = padic(n, base=3)
    val['tc'] = tc3

    col_desc['v3'] = f'the 3-adic valuation of n, such that n = tc * 3^v3'
    val['v3'] = v3

    if arg.odd_core_info:
        paramname = 'oc(n)'
        param = oc
    else:
        paramname = 'n'
        param = n

    col_desc['dist'] = f'the number of odd-to-odd Collatz for {paramname} steps to reach 1'
    val['dist'] = collatz_odd_dist(param)

    col_desc['next'] = f'the next odd-to-odd Collatz step of {paramname}'
    val['next'] = collatz_odd_next(param)

    col_desc['orbit'] = f'the remaining orbit of {paramname} under the odd-to-odd Collatz map, all the way to 1'
    val['orbit'] = collatz_orbit_str(param)

    col_desc['orbit_back'] = f'the remaining orbit of {paramname} under the odd-to-odd Collatz map, bacwards from 1'
    val['orbit_back'] = collatz_orbit_str_backwards(param)

    col_desc['m*'] = f'the prime residues of {paramname} modulo a set of primes'
    if arg.residues > 0:
        for r in range(arg.residues):
            p = PRIMES[r]
            val['m' + str(p)] = param % p

    for radix in [2, 3, 5, 7, 11]:
        col_desc['base' + str(radix)] = f'the {radix_name(radix)} representation of {paramname}'
        in_radix = get_str_in_radix(param, radix)
        val['base' + str(radix)] = in_radix

    for radix in [2, 3]:
        symbol = get_str_in_radix(param, 2)
        remap = symbol_map[radix]
        for i, char in enumerate(remap):
            symbol = symbol.replace(char, remap[i])

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

    if 'factors' in arg.cols or 'rad' in arg.cols or 'mask' in arg.cols:
        all_factors = get_prime_factors(param)
        factors_str = ', '.join([str(f) for f in all_factors])

        cursor_factors = get_prime_factors(param)
        for exclude in [2, 3]:
            if exclude in cursor_factors:
                cursor_factors.remove(exclude)
        cursor_factor_str = ', '.join([str(f) for f in cursor_factors])

        if arg.factor_braces:
            factors_str = '{' + factors_str + '}'
            cursor_factor_str = '{' + cursor_factor_str + '}'

        col_desc['factors'] = f'the prime factors of {paramname}'
        val['factors'] = factors_str

        col_desc['cfactors'] = f'the prime factors above 3 of {paramname}'
        val['cfactors'] = cursor_factor_str

        col_desc['rad'] = f'the radical of {paramname}, the product of its unique prime factors'
        rad = 1
        for p in all_factors:
            rad *= p
        val['rad'] = rad

        col_desc['crad'] = f'the Collatz "cursor radical" of {paramname}, the product of its prime factors above 3'
        crad = 1
        for p in cursor_factors:
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

    oclo, klo = (ocl, kl) if kl < kr else (ocr, kr)
    ochi, khi = (ocr, kr) if kl < kr else (ocl, kl)

    col_desc['ocl'] = f'the odd core of n-1 (the even number to the left of odd n)'
    val['ocl'] = oclo
    col_desc['kl'] = f'the k value n-1 (the even number to the left of odd n)'
    val['kl'] = klo
    col_desc['ocr'] = f'the odd core of n+1 (the even number to the right of odd n)'
    val['ocr'] = ochi
    col_desc['kr'] = f'the k value n+1 (the even number to the right of odd n)'
    val['kr'] = khi
    col_desc['nl'] = f'the odd core valuation of the even number to the left of odd n'
    val['nl'] = f'({ocl}, {kl})'
    col_desc['nr'] = f'the odd core valuation of the even number to the right of odd n'
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
    if n in n_cache:
        return n_cache[n]
    val = calc_n_values(n)
    n_cache[n] = val
    return val

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

    sep = ','

    cols = arg.cols.split(',')
    col_info = {}
    col_headers = []
    for col in cols:
        # remove leading '-'
        col_raw = col.lstrip('-')
        col_info[col_raw] = col_desc[col_raw]

        if col_raw == 'm*':
            if arg.residues > 0:
                for r in range(arg.residues):
                    col_headers.append('m' + str(PRIMES[r]))
        else:
            col_headers.append(col_raw)

    cells = {}
    for col in col_headers:
        cells[col] = []
        if arg.header_line:
            cells[col].append(col + sep)

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
            col_width = max(col_width, len(col))
            col_width += len(sep)
            justified = []
            for cellval in cells[col]:
                if ('-' + col) in arg.cols:
                    justified.append(cellval.rjust(col_width))
                else:
                    justified.append(cellval.ljust(col_width))
            cells[col] = justified

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
            if idx == 0 and arg.header_line:
                continue
            sortlist.append((item, idx))
        sortlist.sort(key=lambda x: x[0])

        sorted_order = []
        if arg.header_line:
            sorted_order.append(0)
        for item, idx in sortlist:
            sorted_order.append(idx)
    else:
        for n in range(len(ns)):
            sorted_order.append(n)


#    for nn in range(len(ns)):
#        n = ns[sorted_order[nn]]
    for n in sorted_order:
        line = ''
        for col in col_headers:
            line += cells[col][n]
        line = line.rstrip()
        if line[-1] == sep.strip():
            if not arg.trailing_comma:
                line = line[:-1]
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


if __name__ == "__main__":
    if arg.command == "table":
        print_table()
    elif arg.command == "orbit":
        print_orbit()
    elif arg.command == "burndown":
        print_burndown()



