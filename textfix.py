#!/usr/bin/env python3
import argparse
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict

Coord = Tuple[int, int]

# Tie-break order for 4-neighbors: Up, Left, Right, Down
DIRS: List[Coord] = [(-1, 0), (0, -1), (0, 1), (1, 0)]

# Ordered 8-neighborhood for deterministic tie-breaks (scan order matters)
DIRS8: List[Coord] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def read_grid(path: str) -> List[List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    width = max((len(line) for line in lines), default=0)
    return [list(line.ljust(width)) for line in lines]

def grid_size(grid: List[List[str]]) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)

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
                    if not in_bounds(nr, nc, R, C):
                        # We skip edges anyway, so all 8 will exist, but keep guard
                        continue
                    v = grid[nr][nc]
                    cnt[v] += 1
                    if v not in first_seen:
                        first_seen[v] = idx

                if not cnt:
                    continue

                max_count = max(cnt.values())
                # Need at least `threshold` neighbors agreeing
                if max_count < threshold:
                    continue

                # Among the tied max neighbors, pick the one seen earliest in DIRS8
                tied = [k for k, v in cnt.items() if v == max_count]
                chosen_neighbor = min(tied, key=lambda ch2: first_seen[ch2])

                if chosen_neighbor != ch:
                    to_change.append((r, c, chosen_neighbor))

        for r, c, v in to_change:
            grid[r][c] = v

        total_changes += len(to_change)

    return total_changes

def write_grid(grid: List[List[str]]) -> str:
    # Keep lines ragged on output (strip trailing spaces for readability)
    return "\n".join("".join(row).rstrip() for row in grid)

def main():
    parser = argparse.ArgumentParser(
        description="Flood fill, despeckle/melt, and pathfinding over a text map."
    )
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--despeckle", dest="despeckle", default="", help="Characters to despeckle (e.g., '░█')")
    parser.add_argument("--melt", dest="melt", default="", help="Characters for fuzzy despeckle, processed in order (e.g., '░█░')")
    parser.add_argument("--fill", dest="fill", default="", help="Characters to use for flood fill (e.g., 'xyz')")
    parser.add_argument("--path", dest="path", default="", help="Characters for vertical pathfinding (e.g., 'abcd')")
    args = parser.parse_args()

    grid = read_grid(args.file)

    # Order: clean first, then mutate, then route
    if args.despeckle:
        do_despeckle(grid, args.despeckle)

    if args.melt:
        do_melt(grid, args.melt, threshold=6)

    if args.fill:
        do_fill(grid, args.fill)

    if args.path:
        do_paths(grid, args.path)

    print(write_grid(grid))

if __name__ == "__main__":
    main()
