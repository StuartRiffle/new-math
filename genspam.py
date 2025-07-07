def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(1000)


def odd_core(n):
    while n % 2 == 0:
        n = n // 2
    return n

def collatz_next(n):
    n =  3 * n + 1
    return odd_core(n)

def get_str_in_radix(n, base):
    """Convert number to string representation in given base"""
    if n == 0:
        return '0'
    digits = ''
    while n > 0:
        digits += f'{n % base}'
        n //= base
    return digits[::-1]

bases = True
length = 30000
depth = 19
#header = 'desc,k,oc'
header = 'oc,k,dist'

if bases:
    header += ',base3,base2'
for i in range(depth):
    header += ',m' + str(PRIMES[i + 1])


def print_vec(n, comment = ''):
    i = n
    k = 0
    while i > 1 and i % 2 == 0:
        i = i // 2
        k += 1
    #msg = comment
    #msg += ',' 
    msg = str(i)
    msg += ',' + str(k)

    msg += ',' + str(odd_to_odd_dist(i))

    if bases:
        msg += ',' + get_str_in_radix(i, 3)
        msg += ',' + get_str_in_radix(i, 2)

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


known = {
    5: 5, 
    7: 16, 
    27: 111, 
    55: 112, 
    73: 115, 
    97: 118, 
    871: 178, 
    6171: 261}


if False:
    by_dist = {}
    for num in range(1, 101):
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
            print_vec(num - 1, comment = f'oc({num}-1)')
            print_vec(num, comment = f"{num} -> {collatz_next(num)}")
            print_vec(num + 1, comment = f'oc({num}+1)')
            print()        
        print()


if True:
    print(header)
    for startpos in known.keys():
        num = startpos
        print(f'## Collatz orbit of {startpos}')
        print()
        while num > 1:
            #print(f'# {num} (dist {odd_to_odd_dist(num)})')
            print_vec(num - 1, comment = f'oc({num}-1)')
            print_vec(num)#, comment = f"dist = {odd_to_odd_dist(num)}")
            print_vec(num + 1, comment = f'oc({num}+1)')
            print()

            num = collatz_next(num)

        print()

if False:
    print(header)
    for num in range(1, length + 1):
        #if num % 2 == 1:
        #    comment = f'dist {odd_to_odd_dist(num)}'
        #else:
        #    comment = f'oc({num})'
        print_vec(num)#, comment = comment)


if False:
    for num in range(1,10001):
        print(f'{odd}')
