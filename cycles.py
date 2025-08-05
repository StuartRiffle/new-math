def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(1000)
ODD_PRIMES = PRIMES[1:]
num_primes = 4


primes = PRIMES[:num_primes]
cycle = 1
for p in primes:
    cycle *= p


def get_factors(n):
    factors = []
    for p in primes:
        if n % p == 0:
            factors.append(p)
        
    return factors

gaps = []
for i in range(cycle):
    n = i + 1
    factors = get_factors(n)
    desc = ''
    for p in primes:
        if p in factors:
            desc += '1'
        else:
            desc += '0'
    if not factors:
        gaps.append(i)
    
    print(n, factors, desc)

back_gaps = [210 - i for i in reversed(gaps) ]
print(gaps)
print(back_gaps)

common_gaps = [i for i in gaps if i in back_gaps]
print(common_gaps)


print('*10101*101*1*101*10101*10101*1*10101*101*1*101*10101*10101*1*10101*101*1*101*10101*10101*1*10101*101*1*101*10101*10101*1*10101*101*1*101*10101*10101*1*10101*101*1*101*10101*10101*1*10101*101*1*101*10101*10101*1'[::-1])

print('*01001*010*1*010*10010*10010*1*01001*010*1*010*10010*10010*1*01001*010*1*010*10010*10010*1*01001*010*1*010*10010*10010*1*01001*010*1*010*10010*10010*1*01001*010*1*010*10010*10010*1*01001*010*1*010*10010*10010*1'[::-1])
print('*00010*001*0*010*00100*01000*1*00010*001*0*010*00100*01000*1*00010*001*0*010*00100*01000*1*00010*001*0*010*00100*01000*1*00010*001*0*010*00100*01000*1*00010*001*0*010*00100*01000*1*00010*001*0*010*00100*01000*1'[::-1])
print('*000001000*0*100*00010*00001*0*00010*000*1*000*01000*00100*0*01000*001*0*000100000*10000*0100000*100*0*010*00001*0000010*00001*000*01000*00100*00010*0*00100*00010*000*10000*01000*0*10000*010*0*001*00000100000*1'[::-1])
ref = '.#.........#.#...#.#...#.....#.#.....#...#.#...#.....#.....#.#.....#...#.#.....#...#.....#.......#...#.#...#.#...#.......#.....#...#.....#.#...#.....#.#.....#.....#...#.#...#.....#.#.....#...#.#...#.#.........#.'
# make foo ref backwards
foo = ref[::-1]

print(ref)
print(foo)
print('fuick')

ref = ref[:-1]
foo = foo[:-1].replace('#', 'o')

print(ref)

for d in range(0, len(foo), 2):
    # Rotate right by d
    snip = (foo + foo)[len(foo)-d::]
    #snip = snip[:len(foo)]


    # Build line by comparing snip and foo
    line = ''
    hits = 0
    for i in range(len(foo)):
        if snip[i] == 'o' and ref[i] == '#':
            line += '#'
            hits += 1
        else:
            line += snip[i]

    print(f'{d:3}: {hits:3} {line}')

