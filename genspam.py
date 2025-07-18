def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(1000000)
odd_primes = PRIMES[1:]


def padic(base, n):
    k = 0
    if n != 0:
        while n % base == 0:
            n = n // base
            k += 1
    return n, k

def oc(n):
    n, _ = padic(2, n)
    return n

def k(n):
    _, k = padic(2, n)
    return k


def odd_core_k(n):
    return padic(2, n)

def calc_stopping_time(n):
    oc_l,  k_l  = padic(2, n-1)
    oc_r,  k_r  = padic(2, n+1)
    oc_lo, k_lo = (oc_l, k_l) if k_l < k_r else (oc_r, k_r)
    oc_hi, k_hi = (oc_r, k_r) if k_l < k_r else (oc_l, k_l)



if oc(n-1) < oc(n+1):
    return oc(n-1), k(n-1)
else:
    return oc(n+1), k(n+1)


def collatz_next(n):
    n =  3 * n + 1
    return oc(n)

def get_neighbors(n):
    n1, k1 = odd_core_k(n - 1)
    n2, k2 = odd_core_k(n + 1)

    if k1 < k2:
        return n1, n2
    else:
        return n2, n1
    
def get_str_in_radix(n, base):
    """Convert number to string representation in given base"""
    if n == 0:
        return '0'
    digits = ''
    while n > 0:
        digits += f'{n % base}'
        n //= base
    return digits[::-1]

def get_factor_mask(n, include_two = False):
    prime_index = 0
    mask = 0

    prime_list = PRIMES
    if not include_two:
        prime_list = prime_list[1:]

    while n > 1:
        if n % prime_list[prime_index] == 0:
            mask |= 1 << prime_index
            n //= prime_list[prime_index]
        else:
            prime_index += 1

    return mask

def get_factor_mask_binary(n, pad = 0, include_two = False):
    mask = get_factor_mask(n, include_two)
    maskstr = get_str_in_radix(mask, 2)
    maskstr = maskstr[::-1]
    while len(maskstr) < pad:
        maskstr += '0'
    return maskstr

bases = True
add_mask = True
show_odd_k = True
length = 210
depth = 0
#header = 'desc,k,oc'
header = 'oc,k,dist'

if bases:
    header += ',base3,base2'
if add_mask:
    header += ',factors'
for i in range(depth):
    header += ',m' + str(PRIMES[i + 1])

def print_vec(n, comment = '', indent = 0):
    i = n
    k = 0
    while i > 1 and i % 2 == 0:
        i = i // 2
        k += 1
    #msg = comment
    #msg += ',' 

    msg = ' ' * indent
    msg += str(i)
    msg += ','
    if show_odd_k or n % 2 == 0:
        msg += str(k)
    msg += ',' + str(odd_to_odd_dist(i))

    if bases:
        msg += ',' + get_str_in_radix(i, 3)
        msg += ',' + get_str_in_radix(i, 2)

    if add_mask:
        msg += ',' + get_factor_mask_binary(i)

    for j in range(depth):
        msg += ',' + str(i % PRIMES[j + 1])

    if comment:
        msg += ' # ' + comment

    print(msg)


def odd_to_odd_dist(n):
    dist = 0
    while n > 1:
        n = collatz_next(n)
        dist += 1
    return dist



def print_tree(n, indent = 0):
    if n < 2:
        return
    
    left, right = get_neighbors(n)

    print_vec(n - 1, comment = f'oc({n}-1)', indent = indent)
    print_vec(n + 1, comment = f'oc({n}+1)', indent = indent)

    if left != n:
        print_tree(left, indent + 4)
    if right != n:
        print_tree(right, indent + 4)

known = {
    871: 178, 
    #6171: 261,
    }


if False:
    by_dist = {}
    for num in range(1, 1001):
        if num % 2 == 0:
            continue
        dist = odd_to_odd_dist(num)
        if dist not in by_dist:
            by_dist[dist] = []
        by_dist[dist].append(num)

    print(header)
    print()

    for dist in reversed(sorted(by_dist.keys())):
        print(f'# Distance {dist}:')
        print()
        for num in by_dist[dist]:
            if num == 0:
                break

            ocl = oc(num - 1)
            ocr = oc(num + 1)

            print_vec(ocl - 1, comment = f'oc(oc({num}-1)-1)')
            print_vec(num - 1, comment = f'oc({num}-1)')
            print_vec(ocl + 1, comment = f'oc(oc({num}-1)+1)')
            print_vec(num, comment = f"{num} -> {collatz_next(num)}")
            print_vec(ocr - 1, comment = f'oc(oc({num}+1)-1)')
            print_vec(num + 1, comment = f'oc({num}+1)')
            print_vec(ocr + 1, comment = f'oc(oc({num}+1)+1)')
            print()        
        print()


if False:
    print(header)
    for startpos in known.keys():
        num = startpos
        print(f'## Collatz orbit of {startpos}')
        print()
        while num > 1:
            print_vec(num)
            ##print(f'# {num} (dist {odd_to_odd_dist(num)})')
            #ocl = odd_core(num - 1)
            #ocr = odd_core(num + 1)
#
            #print_vec(ocl - 1, comment = f'oc(oc({num}-1)-1)')
            #print_vec(num - 1, comment = f'oc({num}-1)')
            #print_vec(ocl + 1, comment = f'oc(oc({num}-1)+1)')
            #print_vec(num, comment =     f"{num} -> {collatz_next(num)}")
            #print_vec(ocr - 1, comment = f'oc(oc({num}+1)-1)')
            #print_vec(num + 1, comment = f'oc({num}+1)')
            #print_vec(ocr + 1, comment = f'oc(oc({num}+1)+1)')
            #print()        
            num = collatz_next(num)

        print()

if True:
    print_tree(135)

if False:
    print(header)
    ternary_width = len(get_str_in_radix(length + 1, 3))
    binary_width = len(get_str_in_radix(length + 1, 2))
    factor_width = len(get_factor_mask_binary(length + 1, include_two = True))
    #for n in range(1, length + 1):
    n = 27
    while n > 1:
        num = n
        oc, k = odd_core_k(num)
        num = oc

        ternary = get_str_in_radix(num, 3)
        while len(ternary) < ternary_width:
            ternary = '0' + ternary
        binary = get_str_in_radix(num, 2)
        while len(binary) < binary_width:
            binary = '0' + binary
        factors = get_factor_mask_binary(num, factor_width, include_two = True)

        factors = factors.replace('0', '-')
        factors = factors.replace('1', 'X')

        # print num right justified in a field of 4
        print(f'{num:4d},{k},{ternary},{binary},{factors}')
        n = collatz_next(n)



if False:
    for num in range(1,10001):
        print(f'{odd}')

def get_special_path(n):
    path = [str(n)]
    while True:
        nlo, nhi = get_neighbors(n)
        if nlo == 0:
            break
        path.append(f"{nlo}({nhi})")
        n = nlo
    return path

if False:
    print(header)
    path_examples = [31, 33, 73, 75, 83, 85, 97, 587, 599, 9895]
    for num in path_examples:

        special_path = ' -> '.join(get_special_path(num) + [str(1)])
        print(f"# Root path {special_path}")
        print()

        n = num
        while True:
            next = collatz_next(n)
            comment = str(n)
            if n != 1:
                comment += f" (Collatz next is {next})"

            print_vec(n - 1, comment = f'oc({n}-1)')
            print_vec(n, comment = comment)
            print_vec(n + 1, comment = f'oc({n}+1)')
            print()

            if n == 1:
                break

            nlo, nhi = get_neighbors(n)
            #if nlo == 1 and nhi == 1:
            #    break

            n = nlo

        print()
