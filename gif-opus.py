import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io
import sys

# Helper function for Collatz sequence
def collatz_sequence(n, max_steps=None):
    seq = [n]
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return seq

# Helper to map color names to RGB (basic set)
def color_to_rgb(name):
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(name)
    except Exception:
        return (0,0,0)

# Drawing function
def draw_grid_frame(cell_states, grid_cols, grid_rows, cell_w, cell_h, grid_on, border, colors, label=None, label_height=0):
    img_w = border*2 + grid_cols*cell_w
    img_h = border*2 + grid_rows*cell_h + (label_height if label else 0)
    img = Image.new("RGB", (img_w, img_h), (32,32,32))  # Dark gray background
    draw = ImageDraw.Draw(img)
    
    # Draw cells
    for (cx, cy), color_idx in cell_states.items():
        color = colors[color_idx]
        px = border + cx * cell_w
        py = border + cy * cell_h
        
        # Add subtle gradient effect for eliminated cells
        if color_idx == 2:  # Eliminated
            # Draw with slight fade effect
            for i in range(cell_w):
                for j in range(cell_h):
                    fade = 0.7 + 0.3 * (i + j) / (cell_w + cell_h)
                    faded_color = tuple(int(c * fade) for c in color)
                    draw.point((px + i, py + j), fill=faded_color)
        else:
            draw.rectangle([px, py, px+cell_w-1, py+cell_h-1], fill=color)
        
        # Add highlight for cursor
        if color_idx == 3:  # Cursor
            draw.rectangle([px, py, px+cell_w-1, py+cell_h-1], outline=(255,255,255), width=2)
    
    # Draw grid lines (subtle)
    if grid_on:
        grid_color = (64, 64, 64)  # Subtle gray
        for x in range(grid_cols+1):
            px = border + x*cell_w
            draw.line([(px, border), (px, border+grid_rows*cell_h)], fill=grid_color, width=1)
        for y in range(grid_rows+1):
            py = border + y*cell_h
            draw.line([(border, py), (border+grid_cols*cell_w, py)], fill=grid_color, width=1)
    
    # Label with better formatting
    if label and label_height > 0:
        py = border + grid_rows*cell_h
        draw.rectangle([0, py, img_w, py+label_height], fill=(20,20,20))
        font_size = min(label_height-6, 24)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Center the text
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (img_w - tw) // 2
        ty = py + (label_height - th) // 2
        draw.text((tx, ty), label, fill=(255,255,255), font=font)
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Collatz Constraint Visualizer - See how orbits are restricted")
    parser.add_argument("--width", type=int, default=50, help="Grid width")
    parser.add_argument("--height", type=int, default=50, help="Grid height")
    parser.add_argument("--pixels", type=int, default=10, help="Cell size in pixels")
    parser.add_argument("--aspect", type=float, default=1.0, help="Cell aspect ratio")
    parser.add_argument("--grid", action="store_true", default=True, help="Show grid lines")
    parser.add_argument("--border", type=int, default=20, help="Border width")
    parser.add_argument("--available-color", type=str, default="#1E3A5F", help="Color for available cells (dark blue)")
    parser.add_argument("--eliminated-color", type=str, default="#8B0000", help="Color for eliminated cells (dark red)")
    parser.add_argument("--cursor-color", type=str, default="#FFD700", help="Color for current position (gold)")
    parser.add_argument("--n", type=int, default=27, help="Starting number")
    parser.add_argument("--frame-time", type=int, default=200, help="Milliseconds per frame")
    parser.add_argument("--hold-frames", type=int, default=3, help="Frames to hold at start/end")
    parser.add_argument("--label", type=int, default=40, help="Label area height")
    parser.add_argument("--output", type=str, default="collatz_constraints.gif", help="Output filename")
    parser.add_argument("--modulus", type=int, default=0, help="Modulus for residue classes (0=auto)")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum Collatz steps to show")
    parser.add_argument("--skip-frames", type=int, default=1, help="Show every Nth frame to speed up")
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
    skip_frames = args.skip_frames
    
    # Use a smaller modulus than the full grid for clearer patterns
    if args.modulus > 0:
        modulus = args.modulus
    else:
        # Default: use a modulus that creates interesting patterns
        modulus = min(width * height // 4, 1000)
        # Round to a nice number
        if modulus > 100:
            modulus = (modulus // 100) * 100

    cell_w = pixels
    cell_h = int(pixels * aspect)

    # More visually distinct colors
    colors = [
        color_to_rgb(args.available_color),   # 0: Available
        color_to_rgb(args.available_color),   # 1: (unused, same as available)
        color_to_rgb(args.eliminated_color),  # 2: Eliminated
        color_to_rgb(args.cursor_color)       # 3: Current position
    ]

    n = args.n
    seq = collatz_sequence(n, args.max_steps)
    visited_residues = set()
    frames = []
    durations = []

    # Initialize grid - map residue classes to grid positions
    cell_states = {}
    residue_to_cell = {}
    
    # Create a mapping of residue classes to grid cells
    for y in range(height):
        for x in range(width):
            residue = (y * width + x) % modulus
            if residue not in residue_to_cell:
                residue_to_cell[residue] = []
            residue_to_cell[residue].append((x, y))
            cell_states[(x, y)] = 0  # All start as available

    # Helper to update all cells for a given residue
    def mark_residue_eliminated(residue):
        if residue in residue_to_cell:
            for cell_pos in residue_to_cell[residue]:
                if cell_states.get(cell_pos) != 3:  # Don't overwrite cursor
                    cell_states[cell_pos] = 2

    # First frame - show initial state
    initial_residue = n % modulus
    visited_residues.add(initial_residue)
    mark_residue_eliminated(initial_residue)
    
    # Mark cursor position (if n is small enough to be in grid)
    cursor_cell = None
    if n < width * height:
        cursor_y = n // width
        cursor_x = n % width
        if cursor_y < height:
            cursor_cell = (cursor_x, cursor_y)
            cell_states[cursor_cell] = 3
    
    img = draw_grid_frame(cell_states, width, height, cell_w, cell_h, grid_on, border, colors, 
                         label=f"Start: n={n} (mod {modulus} = {initial_residue})", label_height=label_height)
    frames.extend([img]*hold_frames)
    durations.extend([args.frame_time]*hold_frames)

    # Process the Collatz sequence
    frame_count = 0
    for i, value in enumerate(seq[1:], 1):  # Skip first value since we already showed it
        current_residue = value % modulus
        
        # Only create a new frame if this is a new residue or we're showing every frame
        if current_residue not in visited_residues or frame_count % skip_frames == 0:
            visited_residues.add(current_residue)
            mark_residue_eliminated(current_residue)
            
            # Update cursor position if value is in grid
            if cursor_cell:
                cell_states[cursor_cell] = 2  # Mark old cursor position as eliminated
            
            cursor_cell = None
            if value < width * height:
                cursor_y = value // width
                cursor_x = value % width
                if cursor_y < height:
                    cursor_cell = (cursor_x, cursor_y)
                    cell_states[cursor_cell] = 3
            
            # Calculate percentage eliminated
            percent_eliminated = (len(visited_residues) / modulus) * 100
            
            # Create frame
            label_text = f"Step {i}: n={value} (mod {modulus} = {current_residue}) | Eliminated: {percent_eliminated:.1f}%"
            img = draw_grid_frame(cell_states, width, height, cell_w, cell_h, grid_on, border, colors, 
                                 label=label_text, label_height=label_height)
            frames.append(img)
            durations.append(args.frame_time)
        
        frame_count += 1

    # Hold final frame
    for _ in range(hold_frames):
        frames.append(frames[-1])
        durations.append(args.frame_time)

    # Add summary frame with statistics
    percent_eliminated = (len(visited_residues) / modulus) * 100
    available_residues = modulus - len(visited_residues)
    
    # Show pattern of eliminated residues
    eliminated_pattern = []
    for i in range(min(20, modulus)):
        if i in visited_residues:
            eliminated_pattern.append(str(i))
    pattern_str = ", ".join(eliminated_pattern[:10])
    if len(eliminated_pattern) > 10:
        pattern_str += "..."
    
    summary_text = f"Final: {len(visited_residues)}/{modulus} residues eliminated ({percent_eliminated:.1f}%) | Pattern: {pattern_str}"
    final_img = draw_grid_frame(cell_states, width, height, cell_w, cell_h, grid_on, border, colors, 
                               label=summary_text, label_height=label_height)
    frames.extend([final_img] * (hold_frames * 2))
    durations.extend([args.frame_time * 2] * (hold_frames * 2))

    # Save the animation
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    
    # Print summary
    print(f"✓ Saved animation to {output}")
    print(f"  Starting number: {n}")
    print(f"  Modulus: {modulus}")
    print(f"  Steps shown: {len(seq)}")
    print(f"  Residues eliminated: {len(visited_residues)} out of {modulus} ({percent_eliminated:.1f}%)")
    print(f"  Available residues: {available_residues}")
    print("\nThis visualization shows how the Collatz sequence is constrained:")
    print("- Each cell represents numbers with the same remainder when divided by", modulus)
    print("- Red cells show 'forbidden' residue classes that can't appear again")
    print("- The sequence is forced into an ever-shrinking set of possibilities")

if __name__ == "__main__":
    main()
