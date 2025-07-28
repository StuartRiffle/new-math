import sympy

def odd_core(n, radix = 2):
    while n > 1 and n % radix == 0:
        n //= radix
    return n

def odd_prime_radical(n):
    return frozenset(p for p in sympy.primefactors(n) if p != 2)

def is_prime(n):
    return sympy.isprime(n)

def collatz_odd_steps(n):
    """Return number of odd-to-odd steps to reach 1 in Collatz sequence."""
    count = 0
    while n != 1:
        n = 3 * n + 1
        while n % 2 == 0 and n != 1:
            n //= 2
        count += 1
    return count

def generate_graphviz(start=1, end=63, odd_label="number"):
    odd_numbers = list(range(start, end+1, 2))
    radical_seen = set()
    radical_nodes = []
    prime_nodes = []
    regular_nodes = []

    # Classify nodes
    for n in odd_numbers:
        if is_prime(n):
            prime_nodes.append(n)
        else:
            radical = odd_prime_radical(n)
            if radical and radical not in radical_seen:
                radical_nodes.append(n)
                radical_seen.add(radical)
            else:
                regular_nodes.append(n)

    def get_label(n):
        if odd_label == "collatz":
            return collatz_odd_steps(n)
        return n

    dot = [
        "digraph G {",
        "  rankdir=TB;",
        "  node [fontsize=30 style=filled fillcolor=white];",
        "",
    ]

    #dot.append("  // Prime")
    #dot.append("  node [shape=circle];")
    #for n in prime_nodes:
    #    dot.append(f"  {n} [label={get_label(n)}];")
    #dot.append("")

    dot.append("  // Radical")
    dot.append("  node [shape=doublecircle];")
    for n in radical_nodes:
        dot.append(f"  {n} [label=<{get_label(n)}>];")
    dot.append("")

    #dot.append("  // Composite")
    #dot.append("  node [shape=square];")
    #for n in regular_nodes:
    #    dot.append(f"  {n} [label=<{get_label(n)}>];")
    #dot.append("")

    radix = 2
    # Edges (edges still use the numbers, not Collatz label)
    dot.append("  // Edges")
    for n in odd_numbers:
        if odd_prime_radical(n) and not is_prime(n):
            for neighbor in [n-1, n+1]:
                core = odd_core(neighbor, radix=radix)
                if odd_prime_radical(core) or is_prime(core):
                    if core != n and start <= core <= end and core % radix == 1:
                        dot.append(f"  {n} -> {core};")
    dot.append("}")

    return "\n".join(dot)

# Example: Collatz stopping distance as label
print(generate_graphviz(1, 2047, odd_label="number"))
