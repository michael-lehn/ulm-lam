"""The running example of the How-To: generalized least squares,
b = (X^T M^-1 X)^-1 X^T M^-1 y, traced by the Python frontend.  Prints the
ulm_lam IR (intermediate representation) that the frontend writes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from ulm import lam  # noqa: E402
from ulm.lam import Matrix, Vector  # noqa: E402


@lam.function
def gls(X: Matrix(2000, 200, "full_rank"), M: Matrix(2000, 2000, "spd"),
        y: Vector(2000)):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y


print(lam.module([gls]), end="")
