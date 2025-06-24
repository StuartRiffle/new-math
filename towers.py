import argparse
import csv
import sys
from math import isqrt

def generate_primes_up_to(n):
    if n < 2:
        return []
    sieve = [True] * (n+1)
    sieve[0:2] = [False, False]
    for i in range(2, isqrt(n)+1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]

def residue_vector(num, all_primes):
    k = 0
    n = num
    while n % 2 == 0 and n > 0:
        n //= 2
        k += 1
    odd_core = n
    vec = []
    for p in all_primes:
        if p > odd_core:
            vec.append("")
        else:
            vec.append(str(odd_core % p))
    return [str(odd_core), str(k)] + vec

def make_rows(start, end):
    def odd_core(n):
        while n % 2 == 0 and n > 0:
            n //= 2
        return n
    max_odd_core = max(odd_core(n) for n in range(start, end+1))
    primes = generate_primes_up_to(max_odd_core)
    rows = []
    for n in range(start, end+1):
        row = [str(n)] + residue_vector(n, primes)
        rows.append(row)
    return rows

def write_csv(rows, sideways=False):
    if not sideways:
        writer = csv.writer(sys.stdout, lineterminator='\n')
        for row in rows:
            writer.writerow(row)
    else:
        columns = list(map(list, zip(*rows)))
        writer = csv.writer(sys.stdout, lineterminator='\n')
        for col in columns:
            writer.writerow(col)

def main():
    parser = argparse.ArgumentParser(description="Generate residue vector map for numbers.")
    parser.add_argument('start', type=int, help='Start of range')
    parser.add_argument('end', type=int, help='End of range (inclusive)')
    parser.add_argument('--sideways', action='store_true', help='Rotate table 90 degrees left')
    args = parser.parse_args()

    rows = make_rows(args.start, args.end)
    write_csv(rows, args.sideways)

if __name__ == "__main__":
    main()
