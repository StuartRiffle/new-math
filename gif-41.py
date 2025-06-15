import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def collatz_sequence(n):
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq

def color_to_rgb(name):
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(name)
    except Exception:
        return (0, 0, 0)

def build_grid_numbers(width, height, odds_only):
    """
    Return list of (n, x, y) for all numbers to cover a width*height grid,
    skipping even numbers if odds_only is True.
    """
    grid = []
    index = 0
    for y in range(height):
        for x in range(width):
            n = y * width + x + 1
            if odds_only and n % 2 == 0:
                continue
            grid.append((n, x, y))
    return grid

def draw_grid(grid_numbers, visited_residues, modulus, cursor_n, width, height, cell_size,
              grid_on, border, colors, label="", label_height=0, odds_only=False, dim_even=False):
    """
    colors: [unknown, disabled/dim, visited, cursor]
    """
    img_w = border * 2 + width * cell_size
    img_h = border * 2 + height * cell_size + (label_height if label else 0)
    img = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Map number n to grid position (x, y)
    number_to_pos = {n: (x, y) for (n, x, y) in grid_numbers}

    # Draw cells
    for n in range(1, width * height + 1):
        # Figure out grid cell
        if n not in number_to_pos:
            # If "odds only", skip even numbers (leave as background)
            continue
        x, y = number_to_pos[n]
        px = border + x * cell_size
        py = border + y * cell_size

        if odds_only:
            if n % 2 == 0:
                cell_state = 1 if dim_even else 0  # disabled or background
            else:
                if (n % modulus) in visited_residues:
                    cell_state = 2  # visited
                else:
                    cell_state = 0  # unknown
                if n == cursor_n:
                    cell_state = 3  # cursor
        else:
            if dim_even and n % 2 == 0:
                cell_state = 1
            elif (n % modulus) in visited_residues:
                cell_state = 2
            else:
                cell_state = 0
            if n == cursor_n:
                cell_state = 3

        draw.rectangle([px, py, px+cell_size-1, py+cell_size-1], fill=colors[cell_state])

    # Draw grid
    if grid_on:
        for x in range(width + 1):
            px = border + x * cell_size
            draw.line([px, border, px, border + height * cell_size], fill=(0, 0, 0), width=1)
        for y in range(height + 1):
            py = border + y * cell_size
            draw.line([border, py, border + width * cell_size, py], fill=(0, 0, 0), width=1)

    # Draw label
    if label and label_height > 0:
        lpy = border + height * cell_size
        draw.rectangle([0, lpy, img_w, lpy + label_height], fill=(0, 0, 0))
        font_size = min(label_height - 4, 32)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        w, h = draw.textsize(label, font=font)
        draw.text(((img_w-w)//2, lpy + (label_height-h)//2), label,
                  fill=(255,255,255), font=font)
    return img

def main():
    parser = argparse.ArgumentParser(description="Collatz Residue Class Visualization")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--pixels", type=int, default=16)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--grid", action="store_true", default=True)
    parser.add_argument("--border", type=int, default=12)
    parser.add_argument("--odds", action="store_true", default=True, help="Show only odd numbers (default on).")
    parser.add_argument("--dim-even", action="store_true", default=False, help="If True, even cells are dimmed.")
    parser.add_argument("--label", type=int, default=32)
    parser.add_argument("--output", type=str, default="collatz.gif")
    parser.add_argument("--n", type=int, default=27)
    parser.add_argument("--frame-time", type=int, default=260)
    parser.add_argument("--hold-frames", type=int, default=6)
    parser.add_argument("--modulus", type=int, default=0, help="Modulus for residue classes (default: width*height)")
    args = parser.parse_args()

    width, height = args.width, args.height
    N = width * height
    modulus = args.modulus if args.modulus > 0 else N
    odds_only = args.odds
    dim_even = args.dim_even

    # Color palette (edit as desired)
    colors = [
        color_to_rgb("slateblue"),        # 0 - unknown
        color_to_rgb("midnightblue"),     # 1 - disabled/dim
        color_to_rgb("springgreen"),      # 2 - visited
        color_to_rgb("gold")              # 3 - cursor/current
    ]

    grid_numbers = build_grid_numbers(width, height, odds_only)
    seq = collatz_sequence(args.n)
    visited_residues = set()
    frames = []
    durations = []

    # Initial frame (nothing visited yet)
    cursor_n = args.n
    img = draw_grid(grid_numbers, visited_residues, modulus, cursor_n, width, height,
                    args.pixels, args.grid, args.border, colors,
                    label=f"n={cursor_n} (start)", label_height=args.label,
                    odds_only=odds_only, dim_even=dim_even)
    frames.extend([img] * args.hold_frames)
    durations.extend([args.frame_time] * args.hold_frames)

    for step, value in enumerate(seq):
        visited_residues.add(value % modulus)
        img = draw_grid(grid_numbers, visited_residues, modulus, value, width, height,
                        args.pixels, args.grid, args.border, colors,
                        label=f"Step {step}: n={value} (mod {modulus} = {value % modulus})", label_height=args.label,
                        odds_only=odds_only, dim_even=dim_even)
        frames.append(img)
        durations.append(args.frame_time)

    # Final summary frame
    percent_eliminated = 100 * len(visited_residues) / modulus
    summary_label = f"{len(visited_residues)}/{modulus} residues visited ({percent_eliminated:.1f}%)"
    img = draw_grid(grid_numbers, visited_residues, modulus, 1, width, height,
                    args.pixels, args.grid, args.border, colors,
                    label=summary_label, label_height=args.label,
                    odds_only=odds_only, dim_even=dim_even)
    frames.extend([img] * (args.hold_frames * 2))
    durations.extend([args.frame_time] * (args.hold_frames * 2))

    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations,
                   optimize=False, loop=0, disposal=2)
    print(f"Saved GIF to {args.output}")
    print(summary_label)

if __name__ == "__main__":
    main()

