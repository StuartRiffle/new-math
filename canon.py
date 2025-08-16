from automaton import *
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
arg = argparse.ArgumentParser()
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
arg = arg.parse_args()

list_of_characters_like_exponents_of_2_and_3_for_squared_and_cubed_etc = '⁰¹²³⁴⁵⁶⁷⁸⁹'


arg.command = 'orbit'

arg.max = 100000
arg.n = 871
#arg.cols = 'n,base2,base3,factors,dist,nl,ocldist,nr,ocrdist,orbit_back'
arg.cols = '-dist,-n,factors,-ocldist,-nl,ofactorsl,-ocrdist,-nr,ofactorsr,-tcn,-base2,-base3,-base5,-base7,m*,-pre_total'
arg.sort_by = 'pre_total'
arg.odds = True
arg.column_info = True
arg.header_line = True
#arg.pad = True
#arg.autocompare = True
arg.factor_braces = True
arg.residues = 12
arg.pad = True
arg.csv = True
#arg.count_factors = True
#arg.radical_only = True


from sympy import primerange, nextprime, isprime
from sympy.ntheory import factorint

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

mask_map = '-X'

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

def get_prime_factors(n):
    factorization = factorint(n)
    return sorted(list(factorization.keys()))

def get_factor_mask(n):
    mask = ''
    for pidx, p in enumerate(PRIMES):
        if pidx > n:
            break
        if n % p == 0:
            mask += mask_map[1]
        else:
            mask += mask_map[0]
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
    for p in get_prime_factors(n):
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






col_desc = {}
unique_factor_sets = {}




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
    factors_str = ', '.join([str(f) for f in all_factors])
    if factors_str not in unique_factor_sets:
        unique_factor_sets[factors_str] = 0
    unique_factor_sets[factors_str] += 1

    cursor_factors = get_prime_factors(param)
    for exclude in [2, 3]:
        if exclude in cursor_factors:
            cursor_factors.remove(exclude)
    cfactors_str = ', '.join([str(f) for f in cursor_factors])

    factorsl_str = ', '.join([str(f) for f in get_prime_factors(param - 1)])
    factorsr_str = ', '.join([str(f) for f in get_prime_factors(param + 1)])

    ofactorsl_str = ', '.join([str(f) for f in get_prime_factors(param - 1) if f % 2 == 1])
    ofactorsr_str = ', '.join([str(f) for f in get_prime_factors(param + 1) if f % 2 == 1])

    if arg.factor_braces:
        factors_str = '{' + factors_str + '}'
        cfactors_str = '{' + cfactors_str + '}'
        factorsl_str = '{' + factorsl_str + '}'
        factorsr_str = '{' + factorsr_str + '}'
        ofactorsl_str = '{' + ofactorsl_str + '}'
        ofactorsr_str = '{' + ofactorsr_str + '}'

    col_desc['factors'] = f'the prime factors of {paramname}'
    val['factors'] = factors_str

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
            col_headers.append(col_raw)

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
            col_width = max(col_width, len(col))
            col_width += len(sep)
            justified = []
            rjust = ('-' + col) in arg.cols
            for cellval in cells[col]:
                if rjust:
                    justified.append(cellval.rjust(col_width))
                else:
                    justified.append(cellval.ljust(col_width))
            cells[col] = justified
            if arg.header_line:
                if arg.pad and header_line:
                    header_line += ' '
                if rjust:
                    header_line += (col + sep).rjust(col_width)
                else:
                    header_line += (col + sep).ljust(col_width)
                


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


if __name__ == "__main__":
    if arg.command == "table":
        print_table()
    elif arg.command == "orbit":
        print_orbit()
    elif arg.command == "burndown":
        print_burndown()
    elif arg.command == "twins":
        print_twins()

