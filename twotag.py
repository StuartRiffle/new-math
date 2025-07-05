def tag2_system(n):
    rules = {'a': 'bc', 'b': 'a', 'c': 'aa'}
    s = 'a' * n
    while len(s) >= 2:
        print(s)
        first = s[0]
        append = rules[first]
        s = s[2:] + append
    if s:
        print(s)

def tag2_system2(n):
    rules = {'a': 'bc', 'b': 'a', 'c': 'aaa'}
    s = 'a' * n
    while len(s) > 1:
        print(s)
        first = s[0]
        append = rules[first]
        s = s[2:] + append
    print(s)  # Print the last 'a'




tag2_system2(19)


# aaaaaaaaaaaaaaaaaaa
# aaaaaaaaaaaaaaaaabc
# aaaaaaaaaaaaaaabcbc
# aaaaaaaaaaaaabcbcbc
# aaaaaaaaaaabcbcbcbc
# aaaaaaaaabcbcbcbcbc
# aaaaaaabcbcbcbcbcbc
# aaaaabcbcbcbcbcbcbc
# aaabcbcbcbcbcbcbcbc
# abcbcbcbcbcbcbcbcbc
# cbcbcbcbcbcbcbcbcbc
# cbcbcbcbcbcbcbcbcaa
# cbcbcbcbcbcbcbcaaaa
# cbcbcbcbcbcbcaaaaaa
# cbcbcbcbcbcaaaaaaaa
# cbcbcbcbcaaaaaaaaaa
# cbcbcbcaaaaaaaaaaaa
# cbcbcaaaaaaaaaaaaaa
# cbcaaaaaaaaaaaaaaaa
# caaaaaaaaaaaaaaaaaa
# aaaaaaaaaaaaaaaaaaa