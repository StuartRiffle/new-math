import argparse
import sys
from PIL import Image, ImageDraw, ImageFont, ImageColor

# Helper function for Collatz sequence (standard definition)
def collatz_sequence(n):
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq

# Helper to map color names to RGB
def color_to_rgb(name):
    try:
        return ImageColor.getrgb(name)
    except ValueError: # More specific exception for invalid color names
        print(f"Warning: Unknown color name '{name}'. Using black instead.", file=sys.stderr)
        return (0,0,0)

# Drawing function to render a single frame of the grid
def draw_grid_frame(cell_map, total_grid_cols, grid_rows_height, cell_w, cell_h, grid_on, border, colors, label=None, label_height=0):
    img_w = border*2 + total_grid_cols*cell_w
    img_h = border*2 + grid_rows_height*cell_h + (label_height if label else 0)
    img = Image.new("RGB", (img_w, img_h), (0,0,0)) # Black background
    draw = ImageDraw.Draw(img)
    
    # Draw cells
    # cell_map is a dictionary: (n_value, (grid_cx, grid_cy)) -> state_code
    for (n_val, (cx, cy)), state_code in cell_map.items():
        color = colors[state_code]
        px = border + cx * cell_w
        py = border + cy * cell_h
        draw.rectangle([px, py, px+cell_w-1, py+cell_h-1], fill=color)
    
    # Draw grid lines
    if grid_on:
        for x in range(total_grid_cols + 1):
            px = border + x*cell_w
            draw.line([px, border, px, border + grid_rows_height*cell_h - 1], fill=(0,0,0))
        for y in range(grid_rows_height + 1):
            py = border + y*cell_h
            draw.line([border, py, border + total_grid_cols*cell_w - 1, py], fill=(0,0,0))
            
    # Draw label if provided
    if label and label_height > 0:
        label_area_top_y = border + grid_rows_height*cell_h
        draw.rectangle([0, label_area_top_y, img_w, label_area_top_y + label_height], fill=(0,0,0)) # Background for label
        
        try:
            # Attempt to use Arial, fallback to default font if not available
            font_size = min(label_height - 4, 28) # Adjust font size to fit label height
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError: # Catch if font file not found (e.g., Arial not on system)
            font = ImageFont.load_default()

        text = str(label)
        # Get bounding box for text to center it
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = (img_w - text_width) // 2
        text_y = label_area_top_y + (label_height - text_height) // 2
        draw.text((text_x, text_y), text, fill=(255,255,255), font=font) # White text
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Collatz Orbit Visualizer - Animated GIF Generator")
    parser.add_argument("--width", type=int, default=32, help="Grid width in cells. (Default: 32)")
    parser.add_argument("--height", type=int, default=32, help="Grid height in cells. (Default: 32)")
    parser.add_argument("--pixels", type=int, default=8, help="Pixel size for each cell. (Default: 8)")
    parser.add_argument("--aspect", type=float, default=1.0, help="Cell aspect ratio (height/width). (Default: 1.0)")
    parser.add_argument("--grid", action="store_true", default=True, help="Draw grid lines. (Default: True)")
    parser.add_argument("--border", type=int, default=10, help="Border size around the grid. (Default: 10)")
    parser.add_argument("--cell-color", type=str, default="blue", help="Color of normal cells. (Default: blue)")
    parser.add_argument("--disabled-color", type=str, default="darkblue", help="Color of initially disabled cells. (Default: darkblue)")
    parser.add_argument("--used-color", type=str, default="lime", help="Color of cells whose residue has been visited by the orbit. (Default: lime)")
    parser.add_argument("--cursor-color", type=str, default="yellow", help="Color of the current number in the orbit. (Default: yellow)")
    parser.add_argument("--n", type=int, default=27, help="Starting number for the Collatz sequence. (Default: 27)")
    parser.add_argument("--frame-time", type=int, default=300, help="Display time for each frame in milliseconds. (Default: 300)")
    parser.add_argument("--even-time", type=int, default=0, help="Optional: display time for frames showing an even number. If 0, uses frame-time. (Default: 0)")
    parser.add_argument("--hold-frames", type=int, default=5, help="Number of times to repeat the first and last frame. (Default: 5)")
    parser.add_argument("--label", type=int, default=30, help="Height of the label area at the bottom. Set to 0 to disable. (Default: 30)")
    parser.add_argument("--output", type=str, default="collatz.gif", help="Output GIF filename. (Default: collatz.gif)")
    parser.add_argument("--odds", action="store_true", default=True, help="Only display odd numbers in the grid. (Default: True)")
    parser.add_argument("--dim-even", action="store_true", default=False, help="Initially dim (disable) all even numbers in the grid. (Default: False)")
    parser.add_argument("--modulus", type=int, default=0, help="Modulus for tracking visited residue classes. If 0, defaults to number of displayed cells. (Default: 0)")
    parser.add_argument("--filter-modulus", type=int, default=0, help="Modulus for initial filtering (disabling cells) based on congruence. Set to 0 to disable this filter. (Default: 0)")
    parser.add_argument("--filter-congruence", type=int, nargs='*', default=[], 
                        help="Space-separated list of congruence classes (mod filter-modulus) to initially disable. E.g., --filter-modulus 4 --filter-congruence 1 2 will disable cells n where n%%4 is 1 or 2. (Default: empty)")
    args = parser.parse_args()

    # --- Configuration and Setup ---
    grid_width_cells = args.width
    grid_height_cells = args.height
    cell_pixels_w = args.pixels
    cell_pixels_h = int(args.pixels * args.aspect)
    
    label_height = args.label if args.label > 0 else 0

    # Define color palette: index 0=normal, 1=disabled, 2=used, 3=cursor
    colors = [
        color_to_rgb(args.cell_color),
        color_to_rgb(args.disabled_color),
        color_to_rgb(args.used_color),
        color_to_rgb(args.cursor_color)
    ]

    # --- Grid Construction ---
    # `cell_elements`: list of (n_value, (grid_cx, grid_cy)) for all cells that will be displayed
    # `num_to_cell_coord`: maps n_value to its (grid_cx, grid_cy) for quick lookup
    cell_elements = [] 
    num_to_cell_coord = {} 
    
    total_grid_cols = 0 # Max columns in any row, for drawing
    num_cells_in_grid = 0 # Total count of displayed cells

    for y in range(grid_height_cells):
        row_col_counter = 0 # Tracks column index for the *displayed* grid
        for x_orig in range(grid_width_cells): # Iterate over ideal grid positions (1 to width*height)
            n_val = y * grid_width_cells + x_orig + 1 # Calculate the number for this position
            
            if args.odds and n_val % 2 == 0:
                continue # Skip even numbers if in odds mode
            
            # Determine actual display coordinates
            grid_cx = row_col_counter if args.odds else x_orig
            grid_cy = y
            
            cell_elements.append((n_val, (grid_cx, grid_cy)))
            num_to_cell_coord[n_val] = (grid_cx, grid_cy)
            row_col_counter += 1
        
        total_grid_cols = max(total_grid_cols, row_col_counter) # Update max columns found
        num_cells_in_grid += row_col_counter # Accumulate total cells

    # --- Initial Cell States (Static Filtering) ---
    # `cell_states`: dictionary mapping `(n_value, (grid_cx, grid_cy))` to its state_code (0=normal, 1=disabled)
    # This dictionary holds the *base state* of each cell, which will be dynamically overridden for 'used'/'cursor'.
    cell_states = {}
    for n_val, coord in cell_elements:
        cell_states[(n_val, coord)] = 0 # All start as 'normal'

    # Apply initial congruence filtering (disabling cells based on --filter-modulus and --filter-congruence)
    if args.filter_modulus > 0 and args.filter_congruence:
        filter_congruence_set = set(args.filter_congruence) # Convert list to set for faster lookup
        for n_val, coord in cell_elements:
            if n_val % args.filter_modulus in filter_congruence_set:
                cell_states[(n_val, coord)] = 1 # Mark as disabled

    # Apply dim_even logic if enabled and not in odds mode
    if args.dim_even and not args.odds:
        for n_val, coord in cell_elements:
            if n_val % 2 == 0 and cell_states[(n_val, coord)] == 0: # Only if not already disabled by congruence filter
                cell_states[(n_val, coord)] = 1

    # --- Collatz Sequence Calculation and Dynamic Visualization ---
    start_n = args.n
    collatz_orbit = collatz_sequence(start_n)
    
    # `visited_residues`: Tracks which `value % modulus` have been encountered in the orbit
    modulus_for_tracking = args.modulus if args.modulus > 0 else num_cells_in_grid
    visited_residues = set()  

    frames = []
    durations = []

    # Prepare the very first frame to show the initial grid state before animation starts
    initial_frame_cell_states = dict(cell_states) # Copy initial states
    if start_n in num_to_cell_coord:
        initial_coord = num_to_cell_coord[start_n]
        initial_frame_cell_states[(start_n, initial_coord)] = 3 # Temporarily mark start_n as cursor
    
    img_first_frame = draw_grid_frame(initial_frame_cell_states, total_grid_cols, grid_height_cells, cell_pixels_w, cell_pixels_h, args.grid, args.border, colors, 
                                     label=f"Start: n={start_n}" if label_height > 0 else None, label_height=label_height)
    frames.extend([img_first_frame] * args.hold_frames)
    durations.extend([args.frame_time] * args.hold_frames)

    # Animate through the Collatz orbit
    current_frame_cell_states = dict(cell_states) # Start with a fresh mutable copy for animation
    for i, current_val in enumerate(collatz_orbit):
        # Update visited residues for the current step
        current_residue = current_val % modulus_for_tracking
        visited_residues.add(current_residue)
        
        # Re-evaluate states for every cell in the grid for the current frame
        for n_val, coord in cell_elements:
            state_code = cell_states[(n_val, coord)] # Start with fixed base state (normal or disabled)
            
            # If the cell's residue has been visited by the orbit, mark it as 'used'
            if (n_val % modulus_for_tracking) in visited_residues:
                state_code = 2 
                
            # If this cell is the current number in the orbit, mark it as 'cursor' (highest priority)
            if n_val == current_val and n_val in num_to_cell_coord:
                state_code = 3 
                
            current_frame_cell_states[(n_val, coord)] = state_code
        
        # Add the current frame to the list
        frame_time_current = args.even_time if args.even_time and current_val % 2 == 0 else args.frame_time
        img_current = draw_grid_frame(current_frame_cell_states, total_grid_cols, grid_height_cells, cell_pixels_w, cell_pixels_h, args.grid, args.border, colors, 
                                     label=f"n={current_val}, res={current_residue} (mod {modulus_for_tracking})" if label_height > 0 else None, 
                                     label_height=label_height)
        frames.append(img_current)
        durations.append(frame_time_current)

    # --- Final Frame and Save ---
    # Hold the final state of the grid
    frames.extend([frames[-1]] * args.hold_frames)
    durations.extend([args.frame_time] * args.hold_frames)

    # Add a summary frame
    percent_residues_visited = (len(visited_residues) / modulus_for_tracking) * 100
    img_summary_frame = draw_grid_frame(current_frame_cell_states, total_grid_cols, grid_height_cells, cell_pixels_w, cell_pixels_h, args.grid, args.border, colors, 
                                     label=f"Visited {len(visited_residues)}/{modulus_for_tracking} unique residues ({percent_residues_visited:.1f}%)" if label_height > 0 else None, 
                                     label_height=label_height)
    frames.append(img_summary_frame)
    durations.append(args.frame_time * 2)

    # Save the generated GIF
    if frames:
        frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
        print(f"Animation saved to {args.output}")
        print(f"Collatz sequence for n={start_n} visited {len(visited_residues)} out of {modulus_for_tracking} residue classes ({percent_residues_visited:.1f}%)")
    else:
        print("No frames were generated. Please check input parameters.")

if __name__ == "__main__":
    main()

