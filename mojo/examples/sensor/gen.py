"""The example of my numerical linear algebra lectures, for Mojo.

An application reads sensor data b in every step and solves A x = b.  As
written close to the algorithm:

    for l in range(steps):
        b = get_sensor_data(l)
        x = inv(A) * b
        react(l, x)

ulm.lam turns this into what an expert writes: the LU (lower-upper)
factorization of A once, before the loop (getrf), and in the loop only
the solve on b, in place (getrs: row swaps and two triangular solves).
This script writes the formula with ulm.lam's Python frontend and
generates the Mojo module sensor_lam.mojo; main.mojo implements the
sensor and react and calls it.

    python3 gen.py        (or: pixi run sensor, in mojo/)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))

from ulm import lam  # noqa: E402
from ulm.lam import Index, Matrix, Vector  # noqa: E402

get_sensor_data = lam.extern("get_sensor_data", [Index], Vector("n"))
react = lam.extern("react", [Index, Vector("n")])


@lam.function
def control(A: Matrix("n", "n"), steps: Index):
    with lam.loop(steps) as L:
        b = get_sensor_data(L.index)
        x = A.inv @ b
        react(L.index, x)


if __name__ == "__main__":
    lam.emit_mojo(control, HERE / "sensor_lam.mojo")
    (HERE / "sensor_lam.mlir").unlink()  # (the IR it was generated from)
    print(f"wrote {HERE / 'sensor_lam.mojo'}")
