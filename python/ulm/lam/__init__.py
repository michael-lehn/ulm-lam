"""ulm.lam: linear algebra in Python, mapped onto ulmBLAS by an MLIR compiler.

    from ulm import lam
    from ulm.lam import Matrix, Vector

    @lam.function
    def gls(X: Matrix(2000, 200, "full_rank"), M: Matrix(2000, 2000, "spd"),
            y: Vector(2000)):
        return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y

    lib = lam.compile(gls)
    b = lib.gls(X, M, y)

See lam.frontend (the language) and lam.runtime (compiling, calling).
"""

from .frontend import (Float, In, Index, InOut, LamError, Matrix, Vector,
                       assume, extern, eye, function, inv, loop, module,
                       update)
from .runtime import InfoError, Library, Workspace, compile

__all__ = [
    "Float", "In", "Index", "InfoError", "InOut", "LamError", "Library",
    "Matrix", "Vector", "Workspace", "assume", "compile", "extern", "eye",
    "function", "inv",
    "loop", "module", "update",
]
