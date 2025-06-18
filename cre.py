
def calc_primes(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n) if sieve[i]]

PRIMES = calc_primes(10000)


def get_primes_less_than(n):
    return [p for p in PRIMES if p < n]


# a prime number iterator
class PrimeIterator:
    def __init__(self):
        self.primes = PRIMES
        self.index = 0

    def __next__(self):
        self.index += 1
        return self.primes[self.index]

# a prime number generator
def prime_generator():
    for p in PRIMES:
        yield p




