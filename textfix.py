#!/usr/bin/env python3
import argparse
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import re
import math

try:
    from PIL import Image, ImageColor
except ImportError as e:
    Image = None
    ImageColor = None

Coord = Tuple[int, int]

# Tie-break order for 4-neighbors: Up, Left, Right, Down
DIRS: List[Coord] = [(-1, 0), (0, -1), (0, 1), (1, 0)]

# Ordered 8-neighborhood for deterministic tie-breaks (scan order matters)
DIRS8: List[Coord] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Sentinel for argparse "was the option provided?"
_ARG_SENTINEL = object()

# ---------- I/O helpers (now line-aware) ----------

def read_lines(path: str) -> List[str]:
    """Read raw lines (without trailing newlines) from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def lines_to_grid(lines: List[str]) -> List[List[str]]:
    """Convert a list of lines to a rectangular grid (space-padded)."""
    width = max((len(line) for line in lines), default=0)
    return [list(line.ljust(width)) for line in lines]

def read_grid(path: str) -> List[List[str]]:
    """Backward-compat: read file straight into a rectangular char grid."""
    return lines_to_grid(read_lines(path))

def grid_size(grid: List[List[str]]) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)

def write_grid(grid: List[List[str]], strip_trailing: bool = True) -> str:
    """Join rows back to lines. By default strips trailing whitespace per line."""
    out_lines: List[str] = []
    for row in grid:
        s = "".join(row)
        if strip_trailing:
            s = s.rstrip()
        out_lines.append(s)
    return "\n".join(out_lines)

def in_bounds(r: int, c: int, R: int, C: int) -> bool:
    return 0 <= r < R and 0 <= c < C

def neighbors4(r: int, c: int, R: int, C: int):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, R, C):
            yield nr, nc

def neighbors8(r: int, c: int, R: int, C: int):
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, R, C):
            yield nr, nc

# ---------- New line-level transforms ----------

def apply_replacements(lines: List[str], pairs: List[Tuple[str, str]]) -> List[str]:
    if not pairs:
        return lines
    out = []
    for line in lines:
        s = line
        for before, after in pairs:
            if before == "":
                continue
            s = s.replace(before, after)
        out.append(s)
    return out

def apply_strip_like(lines: List[str], which: str, chars_opt):
    """
    which ∈ {'strip','lstrip','rstrip'}
    chars_opt: _ARG_SENTINEL (not requested), None (use whitespace), or str (the char set)
    """
    if chars_opt is _ARG_SENTINEL:
        return lines  # option not provided
    use_whitespace = (chars_opt is None)
    out = []
    for s in lines:
        if which == "strip":
            out.append(s.strip() if use_whitespace else s.strip(chars_opt))
        elif which == "lstrip":
            out.append(s.lstrip() if use_whitespace else s.lstrip(chars_opt))
        elif which == "rstrip":
            out.append(s.rstrip() if use_whitespace else s.rstrip(chars_opt))
        else:
            out.append(s)
    return out

def reverse_lines(lines: List[str]) -> List[str]:
    return [s[::-1] for s in lines]

def right_justify_lines(lines: List[str]) -> List[str]:
    """Right-justify each line to the length of the longest line."""
    width = max((len(s) for s in lines), default=0)
    return [s.rjust(width) for s in lines]

def rpad_lines(lines: List[str]) -> List[str]:
    """Pad lines with spaces on the right to match the longest line."""
    width = max((len(s) for s in lines), default=0)
    return [s.ljust(width) for s in lines]

def count_chars_in_lines(lines: List[str]) -> Counter:
    """Count characters in provided lines (newlines excluded)."""
    cnt: Counter[str] = Counter()
    for s in lines:
        cnt.update(s)
    return cnt

# ---------- New: Columns ----------

def wrap_lines_into_columns(
    lines: List[str],
    column_count: Optional[int],
    column_length: Optional[int],
    split_regex: Optional[str],
    equal_width: bool,
    width_increasing: bool,
    margin: int,
) -> List[str]:
    """
    Arrange `lines` into side-by-side columns.

    Exactly one of `column_count` or `column_length` must be provided.
    If `split_regex` is provided, only start NEW columns at a line that matches the regex.
    """
    if not lines:
        return []

    if (column_count is None) == (column_length is None):
        # either both set or both None
        raise ValueError("Provide exactly one of --column-count or --column-length for --columns.")

    pattern = re.compile(split_regex) if split_regex else None
    n = len(lines)

    # Determine cut points (start indices for each column)
    starts: List[int] = [0]

    if column_count is not None:
        target_len = math.ceil(n / max(1, column_count))
        while len(starts) < column_count and starts[-1] < n:
            want = min(starts[-1] + target_len, n)
            if pattern:
                # find the next index >= want that matches
                next_idx = None
                for i in range(want, n):
                    if pattern.search(lines[i]):
                        next_idx = i
                        break
                if next_idx is None:
                    # no further match; stop splitting (rest stays in last column)
                    break
                starts.append(next_idx)
            else:
                if want >= n:
                    break
                starts.append(want)
    else:
        # Fixed height per column
        H = max(1, column_length)
        idx = 0
        while True:
            want = idx + H
            if want >= n:
                break
            if pattern:
                next_idx = None
                for i in range(want, n):
                    if pattern.search(lines[i]):
                        next_idx = i
                        break
                if next_idx is None:
                    # no more valid starts; last column takes remainder
                    break
                starts.append(next_idx)
                idx = next_idx
            else:
                starts.append(want)
                idx = want

    # Build column slices from starts
    starts = sorted(set([s for s in starts if 0 <= s < n]))
    if not starts or starts[0] != 0:
        starts = [0] + starts
    # Ensure last sentinel end
    ends = starts[1:] + [n]

    columns: List[List[str]] = []
    for a, b in zip(starts, ends):
        if a >= b:
            continue
        columns.append(lines[a:b])

    if not columns:
        return []

    # Compute widths
    widths = [max((len(s) for s in col), default=0) for col in columns]

    # Equalize widths if requested
    if equal_width:
        w = max(widths) if widths else 0
        widths = [w for _ in widths]

    # Enforce nondecreasing widths if requested
    if width_increasing and widths:
        for i in range(1, len(widths)):
            if widths[i] < widths[i - 1]:
                widths[i] = widths[i - 1]

    # Render row-wise
    height = max(len(col) for col in columns)
    space = " " * max(0, margin)

    out_lines: List[str] = []
    for r in range(height):
        parts: List[str] = []
        for col_idx, col in enumerate(columns):
            cell = col[r] if r < len(col) else ""
            parts.append(cell.ljust(widths[col_idx]))
        out_lines.append(space.join(parts).rstrip())
    return out_lines

# ---------- Existing grid algorithms ----------

def flood_fill_from_seed(grid: List[List[str]], seed: Coord, target_char: str, replace_char: str) -> None:
    """Fill the connected region of target_char containing seed with replace_char (4-connectivity)."""
    R, C = grid_size(grid)
    sr, sc = seed
    if not in_bounds(sr, sc, R, C):
        return
    if grid[sr][sc] != target_char or target_char == replace_char:
        return

    q = deque([(sr, sc)])
    grid[sr][sc] = replace_char
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c, R, C):
            if grid[nr][nc] == target_char:
                grid[nr][nc] = replace_char
                q.append((nr, nc))

def do_fill(grid: List[List[str]], chars: str) -> None:
    """For each occurrence of each char, choose most frequent 4-neighbor char (stable tie-break)
    and flood-fill that contiguous neighbor region to the char."""
    R, C = grid_size(grid)
    positions: Dict[str, List[Coord]] = {ch: [] for ch in chars}
    for r in range(R):
        for c in range(C):
            ch = grid[r][c]
            if ch in positions:
                positions[ch].append((r, c))

    for ch in chars:
        for (r, c) in positions[ch]:
            counts: Counter[str] = Counter()
            first_seen_dir_for_char: Dict[str, int] = {}
            neigh_coords = []
            for idx, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc, R, C):
                    v = grid[nr][nc]
                    neigh_coords.append((idx, nr, nc, v))
                    counts[v] += 1
                    if v not in first_seen_dir_for_char:
                        first_seen_dir_for_char[v] = idx
            if not counts:
                continue
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            chosen_char = min(candidates, key=lambda cc: first_seen_dir_for_char[cc])

            # Seed = first neighbor in DIRS order that matches chosen_char
            seed: Optional[Coord] = None
            for idx, nr, nc, v in neigh_coords:
                if v == chosen_char:
                    seed = (nr, nc)
                    break
            if seed is not None:
                flood_fill_from_seed(grid, seed, target_char=chosen_char, replace_char=ch)

def bfs_path_on_char(grid: List[List[str]], start: Coord, goal: Coord, terrain: str) -> Optional[List[Coord]]:
    """4-way BFS constrained to cells == terrain."""
    R, C = grid_size(grid)
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nr, nc in neighbors4(r, c, R, C):
            if (nr, nc) not in prev and grid[nr][nc] == terrain:
                prev[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None

def do_paths(grid: List[List[str]], chars: str) -> None:
    """For each char (appearing exactly twice), route vertically on matching background between them."""
    R, C = grid_size(grid)
    for ch in chars:
        locs: List[Coord] = []
        for r in range(R):
            for c in range(C):
                if grid[r][c] == ch:
                    locs.append((r, c))
        if len(locs) != 2:
            continue

        (r1, c1), (r2, c2) = sorted(locs)  # top then bottom
        top, bot = (r1, c1), (r2, c2)

        under_top = (top[0] + 1, top[1])
        above_bot = (bot[0] - 1, bot[1])
        if not (in_bounds(*under_top, R, C) and in_bounds(*above_bot, R, C)):
            continue
        bg1 = grid[under_top[0]][under_top[1]]
        bg2 = grid[above_bot[0]][above_bot[1]]
        if bg1 != bg2:
            continue

        terrain = bg1
        path = bfs_path_on_char(grid, under_top, above_bot, terrain)
        if not path:
            continue

        for (r, c) in path:
            grid[r][c] = ch

def do_despeckle(grid: List[List[str]], chars: str) -> int:
    """
    Remove isolated pixels: if center is in `chars` and all 8 neighbors exist and
    are identical (to each other, not the center), replace center with that neighbor.
    One simultaneous pass over all provided chars. Edges are skipped.
    """
    R, C = grid_size(grid)
    to_change: List[Tuple[int, int, str]] = []

    for r in range(1, R - 1):
        for c in range(1, C - 1):
            center = grid[r][c]
            if center not in chars:
                continue
            neigh_vals = [grid[nr][nc] for nr, nc in neighbors8(r, c, R, C)]
            if len(neigh_vals) == 8 and len(set(neigh_vals)) == 1:
                neigh_char = neigh_vals[0]
                if neigh_char != center:
                    to_change.append((r, c, neigh_char))

    for r, c, v in to_change:
        grid[r][c] = v

    return len(to_change)

def do_melt(grid: List[List[str]], chars: str, threshold: int = 5) -> int:
    """
    Fuzzy despeckle: for each char in `chars` (processed in order), replace any
    interior occurrence where at least `threshold` of its 8 neighbors are the same
    character with that neighbor character. Each char is applied as a simultaneous pass.
    Deterministic tie-break among equally-common neighbor chars by first-seen in DIRS8.
    """
    R, C = grid_size(grid)
    total_changes = 0

    for ch in chars:
        to_change: List[Tuple[int, int, str]] = []

        for r in range(1, R - 1):
            for c in range(1, C - 1):
                if grid[r][c] != ch:
                    continue

                # Count neighbors and remember first-seen order index per char
                cnt: Counter[str] = Counter()
                first_seen: Dict[str, int] = {}
                for idx, (dr, dc) in enumerate(DIRS8):
                    nr, nc = r + dr, c + dc
                    v = grid[nr][nc]
                    cnt[v] += 1
                    if v not in first_seen:
                        first_seen[v] = idx

                max_count = max(cnt.values())
                if max_count < threshold:
                    continue

                tied = [k for k, v in cnt.items() if v == max_count]
                chosen_neighbor = min(tied, key=lambda ch2: first_seen[ch2])

                if chosen_neighbor != ch:
                    to_change.append((r, c, chosen_neighbor))

        for r, c, v in to_change:
            grid[r][c] = v

        total_changes += len(to_change)

    return total_changes

# ---------- Dumb-simple preprocess ----------

def preprocess_file(path: Path) -> list[str]:
    out: list[str] = []
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.lstrip().startswith("#include"):
                # grab everything after #include
                rest = s.lstrip()[len("#include"):].strip()
                rest = rest.strip('"').strip()
                inc_path = (base_dir / rest)
                with inc_path.open("r", encoding="utf-8") as inc:
                    out.extend(inc.read().splitlines())
            else:
                out.append(s)
    return out

# ---------- Bitmap export ----------

def _parse_color(color_str: str) -> Tuple[int, int, int, int]:
    if ImageColor is None:
        raise RuntimeError("Pillow is required for --bitmap (pip install Pillow).")
    try:
        # Returns RGBA for names and hex (including #RRGGBBAA)
        return ImageColor.getcolor(color_str, "RGBA")
    except Exception as e:
        raise ValueError(f"Invalid color {color_str!r}: {e}") from e

def _parse_bitmap_map(spec: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Parse "a=red,0=#00ff00" into { 'a': (r,g,b,a), '0': (...)}.
    Keys should be single characters. Special keys: 'space' -> ' ', 'tab' -> '\\t'.
    """
    mapping: Dict[str, Tuple[int, int, int, int]] = {}
    if not spec:
        return mapping
    pairs = [p for p in spec.split(",") if p]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--bitmap mapping entry must be CHAR=COLOR, got {pair!r}")
        key, color = pair.split("=", 1)
        key = key.strip()
        color = color.strip()
        if key.lower() == "space":
            ch = " "
        elif key.lower() == "tab":
            ch = "\t"
        else:
            if len(key) != 1:
                raise ValueError(f"--bitmap mapping keys must be single characters (or 'space'/'tab'); got {key!r}")
            ch = key
        mapping[ch] = _parse_color(color)
    return mapping

_DEFAULT_PALETTE = [
    "#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]

def write_bitmap_png(grid: List[List[str]], map_spec: str, out_path: Path, bg_color_str: str) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for --bitmap (pip install Pillow).")

    R, C = grid_size(grid)
    if R == 0 or C == 0:
        # Nothing to write; create a 1x1 transparent image.
        img = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        img.save(out_path)
        return

    user_map = _parse_bitmap_map(map_spec) if map_spec else {}
    bg_rgba = _parse_color(bg_color_str)

    # Build color map in appearance order for any characters not explicitly mapped.
    palette_iter = iter(_parse_color(c) for c in _DEFAULT_PALETTE)
    auto_map: Dict[str, Tuple[int, int, int, int]] = {}

    img = Image.new("RGBA", (C, R), bg_rgba)
    px = img.load()

    for r in range(R):
        for c in range(C):
            ch = grid[r][c]
            if ch == " " and " " not in user_map:
                rgba = bg_rgba  # background for padding/space
            elif ch in user_map:
                rgba = user_map[ch]
            else:
                if ch not in auto_map:
                    try:
                        rgba = next(palette_iter)
                    except StopIteration:
                        # If we run out, start over (still deterministic)
                        palette_iter = iter(_parse_color(c) for c in _DEFAULT_PALETTE)
                        rgba = next(palette_iter)
                    auto_map[ch] = rgba
                else:
                    rgba = auto_map[ch]
            px[c, r] = rgba

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Flood fill, despeckle/melt, pathfinding, line transforms, columns, and bitmap export over a text map."
    )
    parser.add_argument("file", help="Input text file")

    # Existing ops
    parser.add_argument("--despeckle", dest="despeckle", default="", help="Characters to despeckle (e.g., '░█')")
    parser.add_argument("--melt", dest="melt", default="", help="Characters for fuzzy despeckle, processed in order (e.g., '░█░')")
    parser.add_argument("--fill", dest="fill", default="", help="Characters to use for flood fill (e.g., 'xyz')")
    parser.add_argument("--path", dest="path", default="", help="Characters for vertical pathfinding (e.g., 'abcd')")
    parser.add_argument("--preprocess", action="store_true", help='Process #include declarations')

    # Reporting
    parser.add_argument("--countchars", action="store_true",
                        help="List each unique character and its count (after line-level transforms), then exit.")

    # Line transforms
    parser.add_argument("--revline", action="store_true",
                        help="Reverse characters in every line (left-to-right).")
    parser.add_argument("--rjust", action="store_true",
                        help="Right-justify all lines to the longest line (pads on the left).")
    parser.add_argument("--rpad", action="store_true",
                        help="Pad lines with spaces on the right so all have equal length (preserves trailing spaces on output).")

    parser.add_argument("--strip", nargs="?", const=None, default=_ARG_SENTINEL, metavar="CHARS",
                        help="Strip characters from both ends. With no CHARS, strips whitespace; with CHARS, strips those characters.")
    parser.add_argument("--lstrip", nargs="?", const=None, default=_ARG_SENTINEL, metavar="CHARS",
                        help="Strip characters from the left. With no CHARS, strips whitespace; with CHARS, strips those characters.")
    parser.add_argument("--rstrip", nargs="?", const=None, default=_ARG_SENTINEL, metavar="CHARS",
                        help="Strip characters from the right. With no CHARS, strips whitespace; with CHARS, strips those characters.")

    parser.add_argument("--replace", action="append", default=[], metavar="BEFORE=AFTER",
                        help="Simple string replacement (literal). May be repeated; applied in order per line.")

    # Columns
    parser.add_argument("--columns", action="store_true",
                        help="Wrap lines into side-by-side columns.")
    parser.add_argument("--column-count", type=int, default=None,
                        help="Target number of columns (use with --columns).")
    parser.add_argument("--column-length", type=int, default=None,
                        help="Target number of lines per column (use with --columns).")
    parser.add_argument("--column-equal-width", action="store_true",
                        help="Pad all columns to the same width.")
    parser.add_argument("--column-width-increasing", action="store_true",
                        help="Ensure column widths are nondecreasing left-to-right.")
    parser.add_argument("--column-split-regex", default=None, metavar="REGEX",
                        help="Only start NEW columns at a line matching REGEX.")
    parser.add_argument("--column-margin", type=int, default=1,
                        help="Spaces between columns (default 1).")

    # Bitmap export
    parser.add_argument("--bitmap", nargs="?", const="", metavar="MAP",
                        help="Write a PNG where each character is one pixel. "
                             "Optionally provide MAP like \"0=black,1=red,A=#00ff00\". "
                             "If MAP omitted, colors are auto-assigned.")
    parser.add_argument("--bitmap-out", default=None, metavar="PATH",
                        help="Output PNG path (defaults to <input_stem>.png).")
    parser.add_argument("--bgcolor", default="#ffffff00", metavar="COLOR",
                        help="Background color for spaces (named or #RRGGBB[#AA]). Default transparent white.")

    args = parser.parse_args()

    # --- Read raw lines and apply line-level transforms ---
    lines = read_lines(args.file)

    if args.preprocess:
        lines = preprocess_file(Path(args.file))

    # Parse replacements BEFORE other transforms.
    if args.replace:
        pairs: List[Tuple[str, str]] = []
        for spec in args.replace:
            if "=" not in spec:
                parser.error(f"--replace expects BEFORE=AFTER, got {spec!r}")
            before, after = spec.split("=", 1)
            pairs.append((before, after))
        lines = apply_replacements(lines, pairs)

    # Stripping
    lines = apply_strip_like(lines, "lstrip", args.lstrip)
    lines = apply_strip_like(lines, "rstrip", args.rstrip)
    lines = apply_strip_like(lines, "strip",  args.strip)

    # Justify / reverse / pad
    if args.rjust:
        lines = right_justify_lines(lines)
    if args.revline:
        lines = reverse_lines(lines)
    if args.rpad:
        lines = rpad_lines(lines)

    # Columns (after line-level transforms, before grid algorithms)
    if args.columns:
        try:
            lines = wrap_lines_into_columns(
                lines=lines,
                column_count=args.column_count,
                column_length=args.column_length,
                split_regex=args.column_split_regex,
                equal_width=args.column_equal_width,
                width_increasing=args.column_width_increasing,
                margin=args.column_margin,
            )
        except ValueError as e:
            parser.error(str(e))

    # --- Reporting-only mode ---
    if args.countchars:
        cnt = count_chars_in_lines(lines)
        for ch, n in sorted(cnt.items(), key=lambda kv: (-kv[1], ord(kv[0]))):
            print(f"{repr(ch)}\t{n}")
        return

    # --- Convert to grid and run grid-level algorithms ---
    grid = lines_to_grid(lines)  # pads with spaces to max width (background)

    # Order: clean first, then mutate, then route
    if args.despeckle:
        do_despeckle(grid, args.despeckle)

    if args.melt:
        do_melt(grid, args.melt, threshold=6)

    if args.fill:
        do_fill(grid, args.fill)

    if args.path:
        do_paths(grid, args.path)

    # Bitmap export (after all mutations)
    if args.bitmap is not None:
        # Decide output path
        out_path = Path(args.bitmap_out) if args.bitmap_out else Path(args.file).with_suffix(".png")
        try:
            write_bitmap_png(grid, map_spec=args.bitmap, out_path=out_path, bg_color_str=args.bgcolor)
        except Exception as e:
            parser.error(str(e))

    # Preserve trailing spaces if user intentionally padded with --rpad
    strip_trailing = not args.rpad
    print(write_grid(grid, strip_trailing=strip_trailing))

if __name__ == "__main__":
    main()

