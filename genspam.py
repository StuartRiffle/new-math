def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(1000)

def get_str_in_radix(n, base):
    """Convert number to string representation in given base"""
    if n == 0:
        return '0'
    digits = ''
    while n > 0:
        digits += f'{n % base}'
        n //= base
    return digits[::-1]

bases = False
length = 10000
depth = 20
header = 'n,k,oc'

if bases:
    header += ',base3,base2'
for i in range(depth):
    header += ',m' + str(PRIMES[i + 1])
print(header)
for n in range(1, length + 1):
    msg = str(n)
    i = n
    k = 0
    while i > 1 and i % 2 == 0:
        i = i // 2
        k += 1
    msg += ',' + str(k)
    msg += ',' + str(i)
    if bases:
        msg += ',' + get_str_in_radix(i, 3)
        msg += ',' + get_str_in_radix(i, 2)
    for j in range(depth):
        msg += ',' + str(i % PRIMES[j + 1])
    print(msg)


