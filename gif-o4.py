#!/usr/bin/env python3
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageColor

def v2(n):
    """Number of times 2 divides n (i.e. v₂(n))."""
    return (n & -n).bit_length() - 1

def collatz_sequence(n):
    """Return the full Collatz orbit from n down to 1."""
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3*n + 1
        seq.append(n)
    return seq

def draw_grid_frame(states, cols, rows, cw, ch, grid, border, colors, label, lh):
    """Render one frame of the grid."""
    img_w = border*2 + cols*cw
    img_h = border*2 + rows*ch + (lh if label else 0)
    img = Image.new("RGB", (img_w, img_h), (0,0,0))
    draw = ImageDraw.Draw(img)

    # draw cells
    for (n,(cx,cy)), st in states.items():
        px = border + cx*cw
        py = border + cy*ch
        draw.rectangle([px,py, px+cw-1,py+ch-1], fill=colors[st])

    # optional grid lines
    if grid:
        for x in range(cols+1):
            gx = border + x*cw
            draw.line([(gx,border),(gx,border+rows*ch)], fill=(0,0,0))
        for y in range(rows+1):
            gy = border + y*ch
            draw.line([(border,gy),(border+cols*cw,gy)], fill=(0,0,0))

    # optional label bar
    if label and lh:
        ty = border + rows*ch
        draw.rectangle([0,ty, img_w,ty+lh], fill=(0,0,0))
        try:
            font = ImageFont.truetype("arial.ttf", min(lh-4,28))
        except:
            font = ImageFont.load_default()
        #tw, th = draw.textlength(label, font=font)
        #draw.text(((img_w-tw)//2, ty+(lh-th)//2), label, fill=(255,255,255), font=font)

    return img

def main():
    p = argparse.ArgumentParser(prog="Collatz CRT Sieve GIF")
    p.add_argument("--width",       type=int,   default=100)
    p.add_argument("--height",      type=int,   default=100)
    p.add_argument("--pixels",      type=int,   default=8)
    p.add_argument("--aspect",      type=float, default=1.0)
    p.add_argument("--grid",        action="store_true", default=True)
    p.add_argument("--border",      type=int,   default=1)
    p.add_argument("--n",           type=int,   default=27,
                   help="starting odd seed")
    p.add_argument("--frame-time",  type=int,   default=200,
                   help="milliseconds per frame")
    p.add_argument("--hold-frames", type=int,   default=5,
                   help="how many repeats of first/last frame")
    p.add_argument("--label-height",type=int,   default=16)
    p.add_argument("--output",      type=str,   default="collatz.gif")
    p.add_argument("--odds",        action="store_true", default=True,
                   help="only show odd n in grid")
    p.add_argument("--dim-even",    action="store_true", default=False,
                   help="grey out even cells from the start")
    p.add_argument("--cell-color",  type=str,   default="blue")
    p.add_argument("--elim-color",  type=str,   default="darkblue")
    p.add_argument("--surv-color",  type=str,   default="green")
    p.add_argument("--cursor-color",type=str,   default="yellow")
    args = p.parse_args()

    # precompute some layout
    W, H = args.width, args.height
    cw = args.pixels
    ch = int(args.pixels * args.aspect)
    cols = (W+1)//2 if args.odds else W
    rows = H

    # color palette: 0=unknown, 1=eliminated, 2=survivor, 3=cursor
    colors = [
        ImageColor.getrgb(args.cell_color),
        ImageColor.getrgb(args.elim_color),
        ImageColor.getrgb(args.surv_color),
        ImageColor.getrgb(args.cursor_color),
    ]

    # build a list of (value, (cx,cy)) for every cell
    grid_list = []
    num_to_cell = {}
    for y in range(H):
        col_idx = 0
        for x in range(W):
            n = y*W + x + 1
            if args.odds and n % 2 == 0:
                continue
            cx = col_idx if args.odds else x
            cy = y
            grid_list.append((n, (cx, cy)))
            num_to_cell[n] = (cx, cy)
            col_idx += 1

    # initial state map: everything unknown (0)
    states = { (n,coord): 0 for n,coord in grid_list }
    # optionally grey out even cells
    if args.dim_even:
        for n,coord in grid_list:
            if n % 2 == 0:
                states[(n,coord)] = 1

    seed = args.n
    seq  = collatz_sequence(seed)

    # We'll keep exactly one CRT condition: x ≡ r (mod M)
    M = 1
    r = seed % M

    frames   = []
    durations= []

    # first frame (no elimination yet)
    label = f"step=0   M={M}"
    img = draw_grid_frame(states, cols, rows, cw, ch,
                          args.grid, args.border,
                          colors, label, args.label_height)
    for _ in range(args.hold_frames):
        frames.append(img)
        durations.append(args.frame_time)

    # walk the orbit
    for step, val in enumerate(seq, start=1):
        # only update CRT on odd iterates
        if val % 2 == 1:
            k = v2(3*val + 1)
            M *= (1 << k)
            r  = seed % M

        # repaint every cell: eliminate if x%M != r
        for n,coord in grid_list:
            if n % M != r:
                st = 1    # eliminated
            else:
                st = 2    # still possible
            # highlight the *current* iterate in yellow
            if n == val and coord == num_to_cell.get(n):
                st = 3
            states[(n,coord)] = st

        label = f"step={step}   M={M}"
        img = draw_grid_frame(states, cols, rows, cw, ch,
                              args.grid, args.border,
                              colors, label, args.label_height)
        frames.append(img)
        durations.append(args.frame_time)

    # hold on the final frame a little
    for _ in range(args.hold_frames):
        frames.append(frames[-1])
        durations.append(args.frame_time)

    # save animated GIF
    frames[0].save(args.output,
                   save_all=True,
                   append_images=frames[1:],
                   duration=durations,
                   loop=0,
                   optimize=False,
                   disposal=2)
    print("Saved animation to", args.output)

if __name__ == "__main__":
    main()

