import argparse
from PIL import Image, ImageDraw, ImageFont

# Collatz sequence
def collatz_sequence(n):
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq

# All unique prime factors (set)
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
    return factors

# Full prime factorization (list with multiplicities)
def prime_factors_full(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    if n > 1:
        factors.append(n)
    return factors

# Color mapping
def color_to_rgb(name):
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(name)
    except Exception:
        return (0,0,0)

# Drawing grid
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

# ---- Disabling Algorithms ----

def disable_multiples_of_value(value, grid_state, dim_even, N, *args):
    to_disable = set()
    if value % 2 == 1:
        for m in range(value, N+1, value):
            if m != value and m in grid_state:
                if not (dim_even and (m%2==0)):
                    to_disable.add(m)
    return to_disable

def disables_only_those_primes(value, grid_state, dim_even, N, prime_factor_sets):
    """
    Disable all numbers (except value) whose set of unique prime factors is
    exactly equal to that of value (and at least one of each).
    E.g. for 15: disables 15, 45, 75, 225, ... (products of 3^n * 5^m, n,m>=1)
    """
    to_disable = set()
    if value % 2 == 1:
        target_primes = prime_factor_sets.get(value, set())
        if not target_primes or len(target_primes) == 0:
            return to_disable
        for m in range(1, N+1):
            if m == value or m not in grid_state:
                continue
            if prime_factor_sets.get(m, set()) == target_primes:
                if not (dim_even and (m % 2 == 0)):
                    to_disable.add(m)
    return to_disable

def get_disabled_cells(strategy, value, grid_state, dim_even, N, prime_factor_sets):
    if strategy == "multiples":
        return disable_multiples_of_value(value, grid_state, dim_even, N)
    elif strategy == "products_of_primes":
        return disables_only_those_primes(value, grid_state, dim_even, N, prime_factor_sets)
    else:
        raise ValueError(f"Unknown disable algorithm: {strategy}")

# ---- Main ----

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
    parser.add_argument("--frame-time", type=int, default=50)
    parser.add_argument("--even-time", type=int, default=0)
    parser.add_argument("--hold-frames", type=int, default=15)
    parser.add_argument("--label", type=int, default=30)
    parser.add_argument("--output", type=str, default="collatz.gif")
    parser.add_argument("--odds", action="store_true", default=False, help="odds only mode (default off)")
    parser.add_argument("--dim-even", action="store_true", default=False, help="if enabled, even numbers are always disabled")
    parser.add_argument("--disable-alg", type=str, default="products_of_primes", choices=["multiples", "products_of_primes"],
                        help="algorithm for disabling: 'multiples' or 'products_of_primes'")
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
    disable_strategy = args.disable_alg

    cell_w = pixels
    cell_h = int(pixels * aspect)

    colors = [
        color_to_rgb(args.cell_color),     # 0: enabled
        color_to_rgb(args.disabled_color), # 1: disabled
        color_to_rgb(args.used_color),     # 2: used
        color_to_rgb(args.cursor_color)    # 3: cursor
    ]

    # Build grid
    cell_map_list = []
    num_to_cell = {}
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
    frames = []
    durations = []

    # Precompute unique prime factor sets for all numbers up to N
    prime_factor_sets = {k: prime_factors(k) for k in range(1, N+1)}

    # All cells enabled (0), unless otherwise
    cell_states = {k:0 for k, v in cell_map_list}

    if dim_even and not odds_mode:
        for m in range(2, N+1, 2):
            if m in cell_states:
                cell_states[m] = 1

    # Hold frame 1: only n is cursor
    grid1 = dict(cell_states)
    if n in grid1:
        grid1[n] = 3
    img = draw_grid_frame({(k, num_to_cell[k]):v for k,v in grid1.items()},
                          total_cols, height, cell_w, cell_h, grid_on, border, colors,
                          label=n, label_height=label_height)
    frames.extend([img]*hold_frames)
    durations.extend([args.frame_time]*hold_frames)

    # Hold frame 2: disable according to strategy for the starting n
    grid2 = dict(cell_states)
    if disable_strategy == "multiples" and n % 2 == 0:
        disable_set = set()
    else:
        disable_set = get_disabled_cells(disable_strategy, n, grid2, dim_even, N, prime_factor_sets)
    for m in disable_set:
        grid2[m] = 1
    if n in grid2:
        grid2[n] = 3
    img = draw_grid_frame({(k, num_to_cell[k]):v for k,v in grid2.items()},
                          total_cols, height, cell_w, cell_h, grid_on, border, colors,
                          label=n, label_height=label_height)
    frames.extend([img]*hold_frames)
    durations.extend([args.frame_time]*hold_frames)

    # Collatz animation
    grid_state = dict(grid2)
    for i, value in enumerate(seq):
        if i > 0:
            last_v = seq[i-1]
            if last_v in grid_state:
                grid_state[last_v] = 2 # used
        if value % 2 == 1:
            disable_set = get_disabled_cells(disable_strategy, value, grid_state, dim_even, N, prime_factor_sets)
            for m in disable_set:
                if m in grid_state and grid_state[m] != 2:
                    grid_state[m] = 1
        frame = dict(grid_state)
        if value in frame:
            frame[value] = 3
        frame_time = args.even_time if args.even_time and value % 2 == 0 else args.frame_time
        img = draw_grid_frame({(k, num_to_cell[k]):v for k,v in frame.items()},
                              total_cols, height, cell_w, cell_h, grid_on, border, colors,
                              label=value if label_height > 0 else None, label_height=label_height)
        frames.append(img)
        durations.append(frame_time)

    # Hold final frame
    for _ in range(hold_frames):
        frames.append(frames[-1])
        durations.append(args.frame_time)

    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    print(f"Saved animation to {output}")

if __name__ == "__main__":
    main()

