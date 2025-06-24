import argparse
import sys

# Define the list of small odd primes for the residue vector
PRIMES = [3, 5, 7, 11, 13, 17, 19]

def get_odd_core(n):
    """
    Calculates the odd core and the 2-adic valuation (k) of a number n.
    The odd core is the result of repeatedly dividing n by 2 until it is odd.
    k is the number of divisions performed.

    Args:
        n (int): The input integer. Must be positive.

    Returns:
        tuple[int, int]: A tuple containing the odd core (oc) and k.
    """
    if n <= 0:
        return (n, 0)
    k = 0
    oc = n
    while oc > 0 and oc % 2 == 0:
        oc //= 2
        k += 1
    return oc, k

def generate_data(start, end):
    """
    Generates the full data grid for the given range of numbers.

    Args:
        start (int): The starting number of the range (inclusive).
        end (int): The ending number of the range (inclusive).

    Returns:
        list[list[int]]: A 2D list where each inner list represents a
                         row for a number n, containing n, oc, k, and
                         the residue vector.
    """
    grid = []
    for n in range(start, end + 1):
        oc, k = get_odd_core(n)
        row = [n, oc, k]
        # Calculate the residue oc % p for each prime in the list
        for p in PRIMES:
            row.append(oc % p)
        grid.append(row)
    return grid

def print_csv(grid, print_header):
    """
    Prints the data grid in a standard CSV format.

    Args:
        grid (list[list[int]]): The 2D data grid.
        print_header (bool): If True, prints the CSV header row.
    """
    if print_header:
        header = ['n', 'oc', 'k'] + [f'm{p}' for p in PRIMES]
        print(','.join(header))
    for row in grid:
        print(','.join(map(str, row)))

def print_sideways(grid):
    """
    Prints the data grid rotated 90 degrees to the left, creating a
    "tower" representation for each number. The columns are aligned for
    readability.

    Args:
        grid (list[list[int]]): The 2D data grid.
    """
    if not grid:
        return
        
    # Convert all data points to strings for width calculation
    str_grid = [[str(item) for item in row] for row in grid]

    # Transpose the grid so rows become columns and vice-versa
    transposed_grid = list(zip(*str_grid))
    
    # Reverse the order of the rows to create the "tower" effect,
    # with n at the bottom.
    transposed_grid.reverse()
    
    # Calculate the maximum width for each column in the transposed view
    # (i.e., for each number's column in the tower representation)
    try:
        col_widths = [max(len(item) for item in col) for col in zip(*transposed_grid)]
    except ValueError: # handles empty grid
        return
    
    header_labels = [f'm{p}' for p in PRIMES] + ['k', 'oc', 'n']
    header_labels.reverse()

    # Print the aligned, transposed data
    for i, row in enumerate(transposed_grid):
        padded_items = [item.rjust(col_widths[j]) for j, item in enumerate(row)]
        print(f"{header_labels[i]:>4} | {' '.join(padded_items)}")


def main():
    """
    Main function to parse arguments and run the generator.
    """
    parser = argparse.ArgumentParser(
        description='Generate a "residue vector" map for a range of integers.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='The starting number of the range (inclusive).'
    )
    parser.add_argument(
        '--end',
        type=int,
        required=True,
        help='The ending number of the range (inclusive).'
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Suppress the header row in the standard CSV output.'
    )
    parser.add_argument(
        '--sideways',
        action='store_true',
        help='Display the table rotated 90 degrees to the left ("tower" view).'
    )

    # In some environments, arguments might be passed differently.
    # This checks for common interactive notebook scenarios.
    if len(sys.argv) == 1:
        # Provide default values if no command-line args are given.
        # e.g., running in a simple Python interpreter.
        # You can change these defaults.
        print("No command-line arguments found. Using default range [1, 20].")
        print("Example command line usage:")
        print("python your_script_name.py --end 50")
        print("python your_script_name.py --start 10 --end 30 --sideways\n")
        args = parser.parse_args(['--end', '20'])
    else:
        args = parser.parse_args()


    if args.start > args.end:
        parser.error("The --start value cannot be greater than the --end value.")

    data = generate_data(args.start, args.end)
    
    if args.sideways:
        print_sideways(data)
    else:
        print_csv(data, not args.no_header)

if __name__ == "__main__":
    main()

