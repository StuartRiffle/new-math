import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io
import sys

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
        return (0,0,0)

def make_grid_indices(width, height, odds_mode):
    grid = []
    if odds_mode:
        cell_count = 0
        for y in range(height):
            for x in range(width):
                n = y*width + x + 1
                if n % 2 == 1:
                    grid.append((n, (x//2, y)))
    else:
        for y in range(height):
            for x in range(width):
                n = y*width + x + 1
                grid.append((n, (x, y)))
    return grid

def draw_grid_frame(cell_map, width, height, cell_w, cell_h, grid_on, border, colors, label=None, label_height=0):
    grid_cols = max((coord[0] for n, coord in cell_map), default=0) + 1 if cell_map else 0
    grid_rows = height
    img_w = border*2 + grid_cols*cell_w
    img_h = border*2 + grid_rows*cell_h + (label_height if label else 0)
    img = Image.new("RGB", (img_w, img_h), (0,0,0))
    draw = ImageDraw.Draw(img)
    for n, (cx, cy) in cell_map:
        color_idx = cell_map[(n, (cx, cy))] if isinstance(cell_map, dict) else 0
        color = colors[color_idx]
        px = border + cx * cell_w
        py = border + cy * cell_h
        draw.rectangle([px, py, px+cell_w-1, py+cell_h-1], fill=color)
    if grid_on:
        for x in range(grid_cols+1):
            px = border + x*cell_w
            draw.rectangle([px, border, px, border+grid_rows*cell_h-1], fill=(0,0,0))
        for y in range(grid_rows+1):
            py = border + y*cell_h
            draw.rectangle([border, py, border+grid_cols*cell_w-1, py], fill=(0,0,0))
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

def can_lead_to_n(m, n):
    return (m % 2 == 0 and m // 2 == n) or (m % 2 == 1 and (m - 1) % 3 == 0 and (m - 1) // 3 == n)

def update_disabled(n, step, disabled, used, grid_state, N, dim_even):
    """
    Update disabled set for the entire board each step based on reverse Collatz constraints
    and a modulo 4 constraint tied to current n's residue.
    """
    mod_base = 4  # Use modulo 4 to create a pattern correlated with n
    mod_n = n % mod_base
    # Possible residues that could lead to mod_n under reverse Collatz
    possible_mods = set()
    for m_mod in range(mod_base):
        m_even = (m_mod * 2) % mod_base
        m_odd = ((m_mod - 1) * 3 if (m_mod - 1) >= 0 else ((m_mod - 1) + mod_base) * 3) % mod_base
        if m_even == mod_n:
            possible_mods.add(m_mod)
        if m_odd == mod_n:
            possible_mods.add(m_mod)
    for m in range(1, N + 1):
        if m not in used and m not in disabled and m in grid_state:
            if not can_lead_to_n(m, n) and m % mod_base not in possible_mods:
                if not (dim_even and (m % 2 == 0)):
                    disabled.add(m)
                    grid_state[m] = 1
    return disabled, grid_state

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

    cell_w = pixels
    cell_h = int(pixels * aspect)

    colors = [color_to_rgb(args.cell_color), color_to_rgb(args.disabled_color), color_to_rgb(args.used_color), color_to_rgb(args.cursor_color)]

    cell_map_list = []
    num_to_cell = dict()
    current_col = 0
    for y in range(height):
        row_col = 0
        for x in range(width):
            n = y * width + x + 1
            if odds_mode and n % 2 == 0:
                continue
            cx = row_col if odds_mode else x
            cy = y
            cell_map_list.append((n, (cx, cy)))
            num_to_cell[n] = (cx, cy)
            row_col += 1
    total_cols = (width + 1) // 2 if odds_mode else width

    N = width * height
    n = args.n
    seq = collatz_sequence(n)
    used = set()
    disabled = set()
    frames = []
    durations = []

    cell_states = {k:0 for k, v in cell_map_list}
    if dim_even and not odds_mode:
        for m in range(2, N+1, 2):
            if m in cell_states:
                cell_states[m] = 1

    grid1 = dict(cell_states)
    if n in grid1:
        grid1[n] = 3
    img = draw_grid_frame({(k, num_to_cell[k]):v for k,v in grid1.items()}, total_cols, height, cell_w, cell_h, grid_on, border, colors, label=n, label_height=label_height)
    frames.extend([img]*hold_frames)
    durations.extend([args.frame_time]*hold_frames)

    grid2 = dict(cell_states)
    if n in grid2:
        grid2[n] = 3
    img = draw_grid_frame({(k, num_to_cell[k]):v for k,v in grid2.items()}, total_cols, height, cell_w, cell_h, grid_on, border, colors, label=n, label_height=label_height)
    frames.extend([img]*hold_frames)
    durations.extend([args.frame_time]*hold_frames)

    grid_state = dict(grid2)
    step = 0
    for i, value in enumerate(seq):
        if i > 0:
            last_v = seq[i-1]
            if last_v in grid_state:
                grid_state[last_v] = 2
                used.add(last_v)
            disabled, grid_state = update_disabled(value, step, disabled, used, grid_state, N, dim_even)
        frame = dict(grid_state)
        if value in frame:
            frame[value] = 3
        frame_time = args.even_time if args.even_time and value % 2 == 0 else args.frame_time
        img = draw_grid_frame({(k, num_to_cell[k]):v for k,v in frame.items()}, total_cols, height, cell_w, cell_h, grid_on, border, colors, label=value if label_height > 0 else None, label_height=label_height)
        frames.append(img)
        durations.append(frame_time)
        step += 1

    for _ in range(hold_frames):
        frames.append(frames[-1])
        durations.append(args.frame_time)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    print(f"Saved animation to {output}")

if __name__ == "__main__":
    main()
