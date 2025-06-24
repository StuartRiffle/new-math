import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_adic_valuation(n, p):
    """
    Calculates the p-adic valuation of n.
    This is the exponent of the highest power of prime p that divides n.
    Returns 0 if n is 0 or not divisible by p.
    """
    if n == 0:
        return 0
    count = 0
    while n > 0 and n % p == 0:
        count += 1
        n //= p
    return count

def get_collatz_odd_sequence(n_start, max_steps=None):
    """
    Generates the sequence of ODD numbers in a Collatz trajectory.

    Args:
        n_start (int): The starting integer.
        max_steps (int, optional): The maximum number of total steps (odd and even)
                                   to compute. Defaults to None (runs until 1).

    Returns:
        list: A list of the odd integers in the sequence.
    """
    if n_start <= 0:
        return []

    sequence = []
    n = n_start
    steps = 0

    while n > 1:
        if max_steps is not None and steps >= max_steps:
            # Also add the final n if we hit max_steps
            if n % 2 != 0:
                sequence.append(n)
            break

        if n % 2 != 0: # If n is odd
            sequence.append(n)
            # Apply the 3n+1 rule
            n = 3 * n + 1
        else: # If n is even
            # Apply the n/2 rule
            n = n // 2
        steps += 1

    # Ensure the final "1" is included if the sequence terminates
    if n == 1 and (max_steps is None or steps < max_steps):
        # The number 1 is the end point of all successful trajectories
        # and has a special place in the phase plot.
        if 1 not in sequence:
             sequence.append(1)

    return sequence

def create_phase_plot(trajectories, output_file, dpi, no_annotations):
    """
    Creates and saves a 2D phase plot of Collatz trajectories.

    Args:
        trajectories (dict): A dictionary mapping start numbers to their trajectories.
        output_file (str): The path to save the output image.
        dpi (int): The resolution of the saved image.
        no_annotations (bool): If True, suppresses text annotations on the plot.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 12))

    # Use a colormap to assign different colors to each trajectory
    colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))

    phi = math.log2(3) # The constant phi ≈ 1.585

    for color, (start_n, odd_sequence) in zip(colors, trajectories.items()):
        # Avoid log(1) which is 0 and can crowd the x-axis
        plot_sequence = [n for n in odd_sequence if n > 1]
        
        # Calculate coordinates for the phase space plot
        # x-axis: Lyapunov function ζ(n)
        # y-axis: log2(n)
        x_coords = [get_adic_valuation(3 * n + 1, 2) - phi * get_adic_valuation(n, 3) for n in plot_sequence]
        y_coords = [math.log2(n) for n in plot_sequence]

        # Plot the trajectory path
        ax.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=4, alpha=0.7, label=f'n = {start_n}', color=color)

        # Add annotations unless suppressed
        if not no_annotations:
            # Annotate the start point
            if x_coords:
                ax.text(x_coords[0], y_coords[0], f'  {start_n}', va='center', ha='left', color=color, fontweight='bold')
   
    # The '1' point is the universal attractor. All paths lead here.
    # Its Lyapunov value is not well-defined in the same way, but we can place it
    # to show the convergence point. ζ(1) = v2(4) - phi*v3(1) = 2.
    ax.plot([2], [0], 'w*', markersize=15, markeredgecolor='black', label='Attractor (n=1)')

    ax.set_title('Collatz Trajectories in Lyapunov Phase Space', fontsize=20)
    ax.set_xlabel('Lyapunov Exponent ζ(n) = v₂(3n+1) - (log₂3)·v₃(n)\n(← More Contraction | More Expansion →)', fontsize=14)
    ax.set_ylabel('Magnitude (log₂ n)', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Plot successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def main():
    """ Main function to parse arguments and run the visualization. """
    parser = argparse.ArgumentParser(
        description='Generate a 2D phase space visualization of Collatz trajectories.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'numbers',
        metavar='N',
        type=int,
        nargs='+',
        help='One or more starting integers for the Collatz sequences.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help='Maximum number of total steps (odd and even) to compute per sequence. \nIf not provided, runs until the sequence reaches 1.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='collatz_phase_space.png',
        help='Output filename for the plot image (e.g., plot.png).'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution of the output image in dots per inch.'
    )
    parser.add_argument(
        '--no_annotations',
        action='store_true',
        help='If set, suppresses the text labels for start points on the plot.'
    )
    
    args = parser.parse_args()

    trajectories = {}
    for n in args.numbers:
        print(f"Generating sequence for n = {n}...")
        sequence = get_collatz_odd_sequence(n, args.max_steps)
        trajectories[n] = sequence

    create_phase_plot(trajectories, args.output, args.dpi, args.no_annotations)

if __name__ == '__main__':
    main()

