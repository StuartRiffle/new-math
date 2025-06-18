def oc(n):
    while n % 2 == 0:
        n = n // 2
    return n    

def L(n):
    n = 3 * n + 1
    return n

def T(n):
    return oc(L(n))
    
def prime_factors(n):
    i = 2
    factors = set()
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.add(i)
    if n > 1:
        factors.add(n)
    return factors    

collatz_test_cases = [19, 27, 871]


def main():
    for start_point in collatz_test_cases:
        available = set(range(1, 1000))
        forbidden = set()
        n = start_point
        
        while n != 1:
            if not n in available:
                print(f"n={n} not in available!")
                break

            if n in forbidden:
                print(f"n={n} in forbidden!")
                break

            disabling = set()
            n_factors = prime_factors(n)
            for a in available:
                a_factors = prime_factors(a)
                if a_factors.issuperset(n_factors):
                    if a % 3 != 0:
                        disabling.add(a)
    
            print(f"n={n} factors={n_factors} disabling={disabling}")

            for x in disabling:
                available.remove(x)
                forbidden.add(x)

            n = T(n)

        print(f"Final:available={available} forbidden={forbidden}")
            

            
   
            

    pass

if __name__ == "__main__":
    main()
