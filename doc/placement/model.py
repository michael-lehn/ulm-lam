"""Cost model for the placement example (doc/placement).

A drone's control loop, per camera frame t:
    image = camera(t)        on the CPU (the camera driver lives there)
    f     = W @ image        a linear layer: features of the image
    x     = A.inv @ f        the state estimate; A is SPD and constant
    u     = K @ x            the controller
    steer(t, u)              on the CPU (the motor driver lives there)
The compiler replaces A.inv by a Cholesky factorisation A = L L^T, moved
out of the loop, and two triangular solves per frame.

Four devices, all numbers made up but of plausible magnitude.  A kernel
on device d takes  latency_d + max(flops / rate, bytes / bandwidth);
a transfer takes  latency + bytes / bandwidth  of the link.  Matrices
are float64.  The placement minimises the total time
    once (factorisation, moving W, L, K)  +  frames * per frame.
"""

import itertools
import math

KB, MB, GB = 1e3, 1e6, 1e9
us = 1e-6

# device: launch latency, flop rate per kernel class (None: cannot run it),
# memory bandwidth, capacity for resident matrices
DEVICES = {
    "CPU":  dict(lat=0.0,   rate={"gemv": 0.1e12, "potrf": 1e12, "trsv": 0.1e12},
                 bw=100 * GB, cache=(8 * MB, 1000 * GB), cap=math.inf),
    "GPU":  dict(lat=10 * us, rate={"gemv": 10e12, "potrf": 20e12, "trsv": 1e12},
                 bw=2000 * GB, cache=None, cap=24 * GB),
    "NPU":  dict(lat=40 * us, rate={"gemv": 50e12, "potrf": None, "trsv": None},
                 bw=8000 * GB, cache=None, cap=4 * GB),
    "FPGA": dict(lat=1 * us, rate={"gemv": 1e12, "potrf": None, "trsv": 1e12},
                 bw=1000 * GB, cache=None, cap=40 * MB),
}

# links: latency, bandwidth (symmetric); missing pairs go through the CPU
LINKS = {
    ("CPU", "GPU"): (10 * us, 25 * GB),
    ("CPU", "NPU"): (20 * us, 16 * GB),
    ("CPU", "FPGA"): (1 * us, 8 * GB),
    ("GPU", "NPU"): (5 * us, 50 * GB),
}


def transfer(a, b, nbytes):
    if a == b:
        return 0.0
    if (a, b) in LINKS or (b, a) in LINKS:
        lat, bw = LINKS.get((a, b)) or LINKS[(b, a)]
        return lat + nbytes / bw
    return transfer(a, "CPU", nbytes) + transfer("CPU", b, nbytes)


def kernel(dev, kind, flops, nbytes, resident=0.0):
    d = DEVICES[dev]
    rate = d["rate"][kind]
    if rate is None or resident > d["cap"]:
        return math.inf
    bw = d["bw"]
    if d["cache"] and nbytes <= d["cache"][0]:
        bw = d["cache"][1]
    return d["lat"] + max(flops / rate, nbytes / bw)


def costs(m, n, k, frames, p):
    """p: device of (gemv W, potrf A, trsv L, gemv K)."""
    dW, dA, dL, dK = p
    W, A, K = 8 * n * m, 8 * n * n, 8 * k * n
    L = A / 2
    # once: move W and K, factorise A, move L to the solver
    once = transfer("CPU", dW, W) + transfer("CPU", dK, K)
    once += transfer("CPU", dA, A) + kernel(dA, "potrf", n**3 / 3, A, A)
    once += transfer(dA, dL, L)
    # per frame
    frame = transfer("CPU", dW, 8 * m)
    frame += kernel(dW, "gemv", 2 * n * m, W, W)
    frame += transfer(dW, dL, 8 * n)
    frame += 2 * kernel(dL, "trsv", n * n, L, L)
    frame += transfer(dL, dK, 8 * n)
    frame += kernel(dK, "gemv", 2 * k * n, K, K)
    frame += transfer(dK, "CPU", 8 * k)
    return once, frame


def best(m, n, k, frames):
    rows = []
    for p in itertools.product(DEVICES, repeat=4):
        once, frame = costs(m, n, k, frames, p)
        rows.append((once + frames * frame, once, frame, p))
    rows.sort()
    return rows


def fmt(t):
    if t == math.inf:
        return "inf"
    if t >= 1:
        return f"{t:.1f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.2f} ms"
    return f"{t * 1e6:.1f} us"


CASES = {
    "small": dict(m=120 * 120, n=900, k=16, frames=100_000),
    "large": dict(m=256 * 256, n=4096, k=16, frames=100_000),
}

if __name__ == "__main__":
    for name, c in CASES.items():
        rows = best(**c)
        total, once, frame, p = rows[0]
        allcpu = [r for r in rows if r[3] == ("CPU",) * 4][0]
        print(f"{name}: image {c['m']}, state {c['n']}, controls {c['k']}, "
              f"frames {c['frames']}")
        print(f"  best  {p}: once {fmt(once)}, per frame {fmt(frame)}, "
              f"total {fmt(total)}")
        print(f"  CPU   {allcpu[3]}: once {fmt(allcpu[1])}, per frame "
              f"{fmt(allcpu[2])}, total {fmt(allcpu[0])}")
        for r in rows[1:4]:
            print(f"  next  {r[3]}: per frame {fmt(r[2])}, total {fmt(r[0])}")


def breakdown(m, n, k, frames, p):
    """The steps of a placement with their times (for the figures)."""
    dW, dA, dL, dK = p
    W, A, K = 8 * n * m, 8 * n * n, 8 * k * n
    L = A / 2
    once = [("move W to " + dW, transfer("CPU", dW, W)),
            ("move K to " + dK, transfer("CPU", dK, K)),
            ("move A to " + dA, transfer("CPU", dA, A)),
            ("potrf on " + dA, kernel(dA, "potrf", n**3 / 3, A, A)),
            ("move L to " + dL, transfer(dA, dL, L))]
    frame = [("image to " + dW, transfer("CPU", dW, 8 * m)),
             ("gemv W on " + dW, kernel(dW, "gemv", 2 * n * m, W, W)),
             ("f to " + dL, transfer(dW, dL, 8 * n)),
             ("2 trsv on " + dL, 2 * kernel(dL, "trsv", n * n, L, L)),
             ("x to " + dK, transfer(dL, dK, 8 * n)),
             ("gemv K on " + dK, kernel(dK, "gemv", 2 * k * n, K, K)),
             ("u to CPU", transfer(dK, "CPU", 8 * k))]
    for title, steps in (("once", once), ("per frame", frame)):
        print(f"  {title}:")
        for what, t in steps:
            if t > 0:
                print(f"    {what:16s} {fmt(t)}")


def scan():
    """The best placement as the size grows (camera side s: m = s*s,
    n = s*s/16), and for the large case as the number of frames grows."""
    print("size scan (frames 100000):")
    last = None
    for s in range(16, 513, 8):
        m, n = s * s, s * s // 16
        p = best(m, n, 16, 100_000)[0][3]
        if p != last:
            print(f"  from camera {s}x{s} (state {n}): {p}")
            last = p
    print("frame scan (large case):")
    last = None
    for e in range(0, 7):
        for f in (1, 2, 5):
            frames = f * 10**e
            p = best(256 * 256, 4096, 16, frames)[0][3]
            if p != last:
                print(f"  from {frames} frames: {p}")
                last = p


if __name__ == "__main__":
    for name, c in CASES.items():
        print(f"{name}, steps of the best placement:")
        breakdown(**c, p=best(**c)[0][3])
    scan()
