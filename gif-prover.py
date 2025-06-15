import argparse

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io
import sys

# Helper function for Collatz sequence
def collatz_sequence(n):
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq

# Helper for finding prime factors
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
    return list(factors)

# Helper to map color names to RGB (basic set)
def color_to_rgb(name):
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(name)
    except Exception:
        return (0,0,0)

def make_grid_indices(width, height, odds_mode):
    """
    Returns a list of (n, (x, y)) tuples where n is the number represented by that cell.
    In odds_mode, only odd numbers are mapped, and the grid has half as many columns.
    """
    grid = []
    if odds_mode:
        cell_count = 0
        for y in range(height):
            for x in range(width):
                n = y*width + x + 1
                if n % 2 == 1:
                    # Determine position in odd grid
                    grid.append((n, (x//2, y)))
    else:
        for y in range(height):
            for x in range(width):
                n = y*width + x + 1
                grid.append((n, (x, y)))
    return grid

# Drawing function
def draw_grid_frame(cell_map, width, height, cell_w, cell_h, grid_on, border, colors, label=None, label_height=0):
    # grid_on is always True if label is present (to keep structure)
    grid_cols = max((coord[0] for n, coord in cell_map), default=0) + 1 if cell_map else 0
    grid_rows = height
    img_w = border*2 + grid_cols*cell_w
    img_h = border*2 + grid_rows*cell_h + (label_height if label else 0)
    img = Image.new("RGB", (img_w, img_h), (0,0,0))
    draw = ImageDraw.Draw(img)
    # Draw cells
    for n, (cx, cy) in cell_map:
        color_idx = cell_map[(n, (cx, cy))] if isinstance(cell_map, dict) else 0
        color = colors[color_idx]
        px = border + cx * cell_w
        py = border + cy * cell_h
        draw.rectangle([px, py, px+cell_w-1, py+cell_h-1], fill=color)
    # Draw grid lines
    if grid_on:
        for x in range(grid_cols+1):
            px = border + x*cell_w
            draw.rectangle([px, border, px, border+grid_rows*cell_h-1], fill=(0,0,0))
        for y in range(grid_rows+1):
            py = border + y*cell_h
            draw.rectangle([border, py, border+grid_cols*cell_w-1, py], fill=(0,0,0))
    # Label
    if label and label_height > 0:
        py = border + grid_rows*cell_h
        draw.rectangle([0, py, img_w, py+label_height], fill=(0,0,0))
        font_size = min(label_height-4, 28)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        text = str(label)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (img_w-tw)//2
        ty = py + (label_height-th)//2
        draw.text((tx, ty), text, fill=(255,255,255), font=font)
    return img

def main():
    parser = argparse.ArgumentParser(description="Collatz Orbit Viewer - Animated GIF Generator")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--aspect", type=float, default=1)
    parser.add_argument("--grid", action="store_true", default=True)
    parser.add_argument("--border", type=int, default=10)
    parser.add_argument("--cell-color", type=str, default="blue")
    parser.add_argument("--disabled-color", type=str, default="darkblue")
    parser.add_argument("--used-color", type=str, default="green")
    parser.add_argument("--cursor-color", type=str, default="yellow")
    parser.add_argument("--n", type=int, default=27)
    parser.add_argument("--frame-time", type=int, default=300)
    parser.add_argument("--even-time", type=int, default=0)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument("--label", type=int, default=30)
    parser.add_argument("--output", type=str, default="collatz.gif")
    parser.add_argument("--odds", action="store_true", default=True, help="odds only mode (default on)")
    parser.add_argument("--dim-even", action="store_true", default=False, help="if enabled, even numbers are always disabled")
    parser.add_argument("--modulus", type=int, default=0, help="modulus for residue classes (defaults to width*height)")
    args = parser.parse_args()

    width = args.width
    height = args.height
    pixels = args.pixels
    aspect = args.aspect
    grid_on = args.grid
    border = args.border
    hold_frames = args.hold_frames
    label_height = args.label if args.label > 0 else 0
    output = args.output
    odds_mode = args.odds
    dim_even = args.dim_even
    
    # Modulus for tracking residue classes (not used in elimination logic here)
    modulus = args.modulus if args.modulus > 0 else (width * height)

    cell_w = pixels
    cell_h = int(pixels * aspect)

    colors = [color_to_rgb(args.cell_color), color_to_rgb(args.disabled_color), 
              color_to_rgb(args.used_color), color_to_rgb(args.cursor_color)]

    # Generate a list of cells and map for drawing
    cell_map_list = [] # List of (n, (cx, cy))
    num_to_cell = dict() # n -> (cx, cy)
    for y in range(height):
        row_col = 0
        for x in range(width):
            n = y * width + x + 1
            if odds_mode and n % 2 == 0:
                continue # skip even numbers
            cx = row_col if odds_mode else x
            cy = y
            cell_map_list.append((n, (cx, cy)))
            num_to_cell[n] = (cx, cy)
            row_col += 1
    total_cols = (width + 1) // 2 if odds_mode else width

    N = width * height
    n = args.n
    seq = collatz_sequence(n)
    visited_odd_parts = set()  # Track visited odd parts in the Collatz sequence
    frames = []
    durations = []

    # Initialize all cells as enabled
    cell_states = {(k, v): 0 for k, v in cell_map_list}

    # Disable even numbers if in normal mode (not odds mode) and dim_even is set
    # (But we may not want this, so we'll ignore dim_even unless it's part of the elimination logic)
    # This is a design choice: we can either disable all even numbers initially or only eliminate them if their odd part is in the Collatz sequence
    if dim_even and not odds_mode:
        for m in range(2, N+1, 2):
            if (m, num_to_cell.get(m)) in cell_states:
                cell_states[(m, num_to_cell[m])] = 1

    # Mark current position
    current_pos = (n, num_to_cell.get(n)) if n in num_to_cell else None
    if current_pos in cell_states:
        cell_states[current_pos] = 3  # Cursor
    
    # First frame
    img = draw_grid_frame(cell_states, total_cols, height, cell_w, cell_h, grid_on, border, colors, 
                         label=f"n={n} mod {modulus}", label_height=label_height)
    frames.extend([img]*hold_frames)
    durations.extend([args.frame_time]*hold_frames)

    # Now run the Collatz sequence
    for i, value in enumerate(seq):
        # Compute the odd part of the current value and add it to the set
        odd_part = value // (value & -value)  # Equivalent to value / 2^v₂(value)
        visited_odd_parts.add(odd_part)
        
        # Reset all cells to base state
        for cell, coord in cell_map_list:
            # Compute the odd part of the cell number
            cell_odd_part = cell // (cell & -cell)
            
            # Default to available
            cell_state = 0
            
            # Mark as eliminated if odd part is visited
            if cell_odd_part in visited_odd_parts:
                cell_state = 2  # Used/eliminated
                
            # Mark current position as cursor
            if cell == value and cell in num_to_cell:
                cell_state = 3  # Cursor
                
            cell_states[(cell, num_to_cell[cell])] = cell_state
        
        # Prepare frame
        frame_time = args.even_time if args.even_time and value % 2 == 0 else args.frame_time
        img = draw_grid_frame(cell_states, total_cols, height, cell_w, cell_h, grid_on, border, colors, 
                             label=f"n={value}, odd_part={odd_part}" if label_height > 0 else None, 
                             label_height=label_height)
        frames.append(img)
        durations.append(frame_time)

    # Hold final frame
    for _ in range(hold_frames):
        frames.append(frames[-1])
        durations.append(args.frame_time)

    # Add summary frame
    percent_eliminated = (len(visited_odd_parts) / (N // 2)) * 100 if odds_mode else (len(visited_odd_parts) / N) * 100
    final_img = draw_grid_frame(cell_states, total_cols, height, cell_w, cell_h, grid_on, border, colors, 
                             label=f"Eliminated {len(visited_odd_parts)} odd parts", 
                             label_height=label_height)
    frames.append(final_img)
    durations.append(args.frame_time * 2)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    print(f"Saved animation to {output}")
    print(f"Visited {len(visited_odd_parts)} odd parts")

if __name__ == "__main__":
    main()
