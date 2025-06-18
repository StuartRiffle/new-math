import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

def collatz_sequence(n, max_steps=1000):
    """Generate the Collatz sequence starting from n, with a safety limit."""
    seq = [n]
    while n != 1 and len(seq) < max_steps:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq

def color_to_rgb(name):
    """Convert color name to RGB tuple."""
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(name)
    except Exception:
        return (0, 0, 0)

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.9
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1: r, g, b = c, x, 0
        elif h < 2: r, g, b = x, c, 0
        elif h < 3: r, g, b = 0, c, x
        elif h < 4: r, g, b = 0, x, c
        elif h < 5: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors.append((r, g, b))
    
    return colors

def draw_grid_frame(cell_map, width, height, cell_w, cell_h, grid_on, border, 
                   colors, residue_colors=None, modulus=None, label=None, label_height=0):
    """Draw a frame of the grid visualization."""
    img_w = border*2 + width*cell_w
    img_h = border*2 + height*cell_h + (label_height if label else 0)
    img = Image.new("RGB", (img_w, img_h), (20, 20, 30))  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Draw cells
    for (n, (cx, cy)), state in cell_map.items():
        # Determine cell color
        if residue_colors and modulus and state != 3:  # Not cursor
            residue = n % modulus
            base_color = residue_colors[residue]
            if state == 0:  # Unused
                color = tuple(max(0, c - 100) for c in base_color)  # Darker version
            elif state == 1:  # Disabled
                color = tuple(c // 3 for c in base_color)  # Much darker
            elif state == 2:  # Used/visited
                color = base_color  # Full brightness
        else:
            color = colors[state]
            
        # Draw the cell
        px = border + cx * cell_w
        py = border + cy * cell_h
        draw.rectangle([px, py, px+cell_w-1, py+cell_h-1], fill=color)
        
        # Add residue label if cells are large enough
        if modulus and cell_w >= 16:
            residue = n % modulus
            font_size = min(cell_h - 4, 12)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            text = str(residue)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x = px + (cell_w - text_w) // 2
            text_y = py + (cell_h - text_h) // 2
            # Text color contrasting with cell color
            brightness = sum(color) / 3 if hasattr(color, '__iter__') else color
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    # Draw grid lines
    if grid_on:
        for x in range(width+1):
            px = border + x*cell_w
            draw.line([(px, border), (px, border+height*cell_h)], fill=(50, 50, 60), width=1)
        for y in range(height+1):
            py = border + y*cell_h
            draw.line([(border, py), (border+width*cell_w, py)], fill=(50, 50, 60), width=1)
    
    # Add label
    if label and label_height > 0:
        py = border + height*cell_h
        draw.rectangle([0, py, img_w, py+label_height], fill=(10, 10, 20))
        font_size = min(label_height-4, 24)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        text = str(label)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (img_w-tw)//2
        ty = py + (label_height-th)//2
        draw.text((tx, ty), text, fill=(220, 220, 220), font=font)
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Collatz Orbit Viewer - Visualize number constraints")
    parser.add_argument("--width", type=int, default=15, help="Width of the grid")
    parser.add_argument("--height", type=int, default=15, help="Height of the grid")
    parser.add_argument("--pixels", type=int, default=30, help="Pixel size of each cell")
    parser.add_argument("--aspect", type=float, default=1, help="Cell aspect ratio")
    parser.add_argument("--grid", action="store_true", default=True, help="Show grid lines")
    parser.add_argument("--border", type=int, default=10, help="Border size")
    parser.add_argument("--n", type=int, default=27, help="Starting number for Collatz sequence")
    parser.add_argument("--frame-time", type=int, default=300, help="Frame duration in ms")
    parser.add_argument("--hold-frames", type=int, default=5, help="Frames to hold at start/end")
    parser.add_argument("--label", type=int, default=30, help="Label height (0 for no label)")
    parser.add_argument("--output", type=str, default="collatz_constraints.gif", help="Output filename")
    parser.add_argument("--odds", action="store_true", default=True, help="Odds only mode (default on)")
    parser.add_argument("--modulus", type=int, default=0, help="Modulus for residue classes (defaults to min(width*height, 32))")
    parser.add_argument("--color-by-residue", action="store_true", default=True, help="Color cells by residue class")
    parser.add_argument("--highlight-factor", type=float, default=1.5, help="Brightness factor for highlighting visited residues")
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
    color_by_residue = args.color_by_residue
    highlight_factor = args.highlight_factor
    
    # Modulus for tracking residue classes
    modulus = args.modulus if args.modulus > 0 else min(width * height, 32)

    cell_w = pixels
    cell_h = int(pixels * aspect)

    # Base colors for cell states
    base_colors = [
        (80, 80, 100),     # 0: Available/unused
        (40, 40, 50),      # 1: Disabled
        (180, 220, 255),   # 2: Used/visited
        (255, 255, 100)    # 3: Cursor
    ]

    # Generate cell mapping
    cell_map_list = []  # List of (n, (cx, cy))
    num_to_cell = {}    # n -> (cx, cy)
    
    # Generate grid positions
    total_cols = width // (2 if odds_mode else 1)  # Half width if odds only
    for y in range(height):
        row_col = 0
        for x in range(width):
            n = y * width + x + 1
            if odds_mode and n % 2 == 0:
                continue  # Skip even numbers in odds mode
            
            cx = row_col if odds_mode else x
            cy = y
            cell_map_list.append((n, (cx, cy)))
            num_to_cell[n] = (cx, cy)
            row_col += 1

    # Generate colors for residue classes
    residue_colors = generate_distinct_colors(modulus) if color_by_residue else None

    # Calculate Collatz sequence
    n = args.n
    seq = collatz_sequence(n)
    visited_residues = set()  # Track visited residue classes

    frames = []
    durations = []

    # Initialize all cells as enabled
    cell_states = {(k, v): 0 for k, v in cell_map_list}  # Initially all available

    # Mark current position
    if n in num_to_cell:
        cell_states[(n, num_to_cell[n])] = 3  # Cursor
    
    # First frame
    img = draw_grid_frame(
        cell_states, total_cols, height, cell_w, cell_h, grid_on, border, 
        base_colors, residue_colors, modulus, 
        label=f"n={n} mod {modulus}", label_height=label_height
    )
    frames.extend([img] * hold_frames)
    durations.extend([args.frame_time] * hold_frames)

    # Run the Collatz sequence
    for i, value in enumerate(seq):
        # Track the current residue class
        current_residue = value % modulus
        visited_residues.add(current_residue)
        
        # Update all cells based on residue class
        for (cell, coord) in cell_map_list:
            cell_residue = cell % modulus
            
            # Default to available
            cell_state = 0
            
            # Mark as visited if residue matches any visited class
            if cell_residue in visited_residues:
                cell_state = 2  # Used/eliminated
                
            # Mark current position as cursor
            if cell == value and cell in num_to_cell:
                cell_state = 3  # Cursor
                
            cell_states[(cell, coord)] = cell_state
        
        # Prepare frame
        img = draw_grid_frame(
            cell_states, total_cols, height, cell_w, cell_h, grid_on, border, 
            base_colors, residue_colors, modulus, 
            label=f"n={value}, residue={current_residue} (mod {modulus})" if label_height > 0 else None, 
            label_height=label_height
        )
        frames.append(img)
        durations.append(args.frame_time)

    # Hold final frame
    for _ in range(hold_frames):
        frames.append(frames[-1])
        durations.append(args.frame_time)

    # Add summary frame
    percent_eliminated = (len(visited_residues) / modulus) * 100
    summary_label = f"Eliminated {len(visited_residues)}/{modulus} residues ({percent_eliminated:.1f}%)"
    final_img = draw_grid_frame(
        cell_states, total_cols, height, cell_w, cell_h, grid_on, border, 
        base_colors, residue_colors, modulus, 
        label=summary_label, 
        label_height=label_height
    )
    frames.append(final_img)
    durations.append(args.frame_time * 2)

    # Save animation
    frames[0].save(
        output, save_all=True, append_images=frames[1:], 
        duration=durations, loop=0, optimize=False, disposal=2
    )
    
    print(f"Saved animation to {output}")
    print(f"Visited {len(visited_residues)} out of {modulus} residue classes ({percent_eliminated:.1f}%)")
    print(f"Visited residues: {sorted(visited_residues)}")

if __name__ == "__main__":
    main()
