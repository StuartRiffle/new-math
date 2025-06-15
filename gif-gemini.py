import argparse
from PIL import Image, ImageDraw, ImageFont
import sys

def collatz_sequence(n):
    """Generates the Collatz sequence for a given starting number n."""
    if n < 1:
        return []
    seq = [n]
    # We use a set for quick checking to detect cycles in non-standard cases
    in_sequence = {n}
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            if n > (sys.maxsize - 1) // 3:
                print(f"Warning: Number {n} is approaching max size. Stopping sequence.")
                break
            n = 3 * n + 1
        
        if n in in_sequence:
            print(f"Warning: Cycle detected at {n}. Stopping sequence.")
            break
        seq.append(n)
        in_sequence.add(n)
    return seq

def color_to_rgb(name):
    """Converts a color name string to an (R, G, B) tuple."""
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(name)
    except ValueError:
        print(f"Warning: Could not parse color '{name}'. Defaulting to black.")
        return (0, 0, 0)

def draw_frame(cell_colors, width, height, cell_size, border, label_text, label_height):
    """Draws a single frame of the grid visualization."""
    img_w = border * 2 + width * cell_size
    img_h = border * 2 + height * cell_size + label_height
    img = Image.new("RGB", (img_w, img_h), (15, 15, 15)) # Dark background
    draw = ImageDraw.Draw(img)

    for (cx, cy), color in cell_colors.items():
        px, py = border + cx * cell_size, border + cy * cell_size
        draw.rectangle([px, py, px + cell_size, py + cell_size], fill=color)

    if label_height > 0:
        label_area_top = border * 2 + height * cell_size
        draw.rectangle([0, label_area_top, img_w, img_h], fill=(0, 0, 0))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, label_height - 10))
        except IOError:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0,0), label_text, font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (img_w - tw) / 2
        ty = label_area_top + (label_height - th) / 2
        draw.text((tx, ty), label_text, fill=(220, 220, 220), font=font)
    return img

def main():
    parser = argparse.ArgumentParser(
        description="A tool to visualize how a Collatz orbit constrains residue classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core settings
    parser.add_argument("--n", type=int, default=27, help="Starting number for the sequence.")
    parser.add_argument("--width", type=int, default=45, help="Grid width in cells.")
    parser.add_argument("--height", type=int, default=25, help="Grid height in cells.")
    parser.add_argument("--modulus", type=int, default=0, help="The modulus M. Defaults to width * height.")
    
    # Animation & Display
    parser.add_argument("--pixels", type=int, default=16, help="Pixel size of each grid cell.")
    parser.add_argument("--frame-time", type=int, default=300, help="Duration of each main frame in ms.")
    parser.add_argument("--even-time", type=int, default=100, help="Shorter duration for even steps for faster pace.")
    parser.add_argument("--hold-frames", type=int, default=10, help="Frames to hold at start/end for emphasis.")
    parser.add_argument("--output", type=str, default="collatz_constraints.gif", help="Output GIF file name.")
    parser.add_argument("--show-all-numbers", action="store_true", help="Show even numbers; grid will be denser with only odds by default.")

    # Colors
    parser.add_argument("--untouched-color", type=str, default="#202020", help="Color for untouched residue classes.")
    parser.add_argument("--visited-color", type=str, default="dodgerblue", help="Color for previously claimed classes.")
    parser.add_argument("--claimed-color", type=str, default="mediumspringgreen", help="Flash color for newly claimed classes.")
    parser.add_argument("--cursor-color", type=str, default="gold", help="Color for the current number's cell.")
    
    args = parser.parse_args()

    # --- Setup ---
    odds_mode = not args.show_all_numbers
    modulus = args.modulus if args.modulus > 0 else (args.width * args.height)
    
    colors = {
        'untouched': color_to_rgb(args.untouched_color),
        'visited': color_to_rgb(args.visited_color),
        'claimed': color_to_rgb(args.claimed_color),
        'cursor': color_to_rgb(args.cursor_color),
    }

    grid_w = args.width if not odds_mode else (args.width + 1) // 2
    grid_h = args.height

    # Map numbers to cell coordinates
    num_to_cell, cell_to_num = {}, {}
    for y in range(grid_h):
        for x in range(grid_w):
            if odds_mode:
                n = (y * grid_w + x) * 2 + 1
            else:
                n = y * grid_w + x + 1
            num_to_cell[n] = (x, y)
            cell_to_num[(x, y)] = n

    seq = collatz_sequence(args.n)
    if not seq:
        print(f"Could not generate sequence for n={args.n}.")
        return

    frames, durations = [], []
    visited_residues = set()

    # --- Animation Loop ---
    for i, value in enumerate(seq):
        current_residue = value % modulus
        is_new_claim = current_residue not in visited_residues
        visited_residues.add(current_residue)

        # Build the color map for the current frame
        cell_colors = {}
        for (cx, cy), n in cell_to_num.items():
            cell_res = n % modulus
            if is_new_claim and cell_res == current_residue:
                cell_colors[(cx, cy)] = colors['claimed']
            elif cell_res in visited_residues:
                cell_colors[(cx, cy)] = colors['visited']
            else:
                cell_colors[(cx, cy)] = colors['untouched']

        # The cursor always on top
        if value in num_to_cell:
            cell_colors[num_to_cell[value]] = colors['cursor']

        # Determine label and frame duration
        if i == 0:
            label = f"Starting at n = {value}"
            duration = args.frame_time * (args.hold_frames // 2)
        else:
            prev_val = seq[i - 1]
            op_text = "n / 2" if prev_val % 2 == 0 else "3n + 1"
            label = f"T({prev_val}) = {value}  |  ({op_text})"
            duration = args.even_time if prev_val % 2 == 0 else args.frame_time
        
        frame = draw_frame(cell_colors, grid_w, grid_h, args.pixels, 5, label, 40)
        frames.append(frame)
        durations.append(duration)

    # --- Final Summary Frame ---
    percent = (len(visited_residues) / modulus) * 100
    label = f"Finished. Visited {len(visited_residues)} of {modulus} classes ({percent:.1f}%)"
    
    # Final frame is just the settled 'visited' colors
    final_cell_colors = {
        (cx, cy): colors['visited'] if (n % modulus) in visited_residues else colors['untouched']
        for (cx, cy), n in cell_to_num.items()
    }
    final_frame = draw_frame(final_cell_colors, grid_w, grid_h, args.pixels, 5, label, 40)
    frames.append(final_frame)
    durations.append(args.frame_time * args.hold_frames)

    print(f"Generating GIF ({len(frames)} frames) for n={args.n}...")
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2, # Important for animations with changing backgrounds/palettes
    )
    print(f"Animation saved to {args.output}")

if __name__ == "__main__":
    main()
