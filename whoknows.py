def k_of(s):
    r = s & 7
    if   r==1: return 2
    elif r==3: return 1
    elif r==5: return 4
    elif r==7: return 1
    else:      raise ValueError(f"s {s} must be odd")

def f_of(s):
    k = k_of(s)
    return (3*s + 1) >> k

def D(s):
    if s == 1:
        return 0
    k = k_of(s)
    return D(f_of(s)) + k + 1


# for every odd up to 1000
for n in range(1, 1001):
    if n % 2 == 1:
        print(n, D(n))
