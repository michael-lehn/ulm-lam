//===- UlmLamKernelOps.td - ulm_lamk dialect operations ----*- tablegen -*-===//
//
// One op per BLAS/LAPACK kernel.  Every op is in destination-passing style
// (DPS): the operand that BLAS overwrites is the `outs` operand.  On tensors
// the op returns the updated value; on memrefs it has no results.  This is
// exactly what One-Shot Bufferize needs to decide in-place updates (trsm
// overwrites B, potrf overwrites A) and to insert copies only when an input
// is still live -- the "overwriting" analysis that Linnea does by hand.
//
// Vectors are n x 1 / 1 x n rank-2 shaped values; the runtime derives
// increments from the strides.
//
//===----------------------------------------------------------------------===//

#ifndef ULM_LAM_KERNEL_OPS_TD
#define ULM_LAM_KERNEL_OPS_TD

include "ulm_lam/UlmLamBase.td"
include "ulm_lam/UlmLamAttrs.td"
include "ulm_lam/UlmLamInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

def UlmLamk_Float : AnyTypeOf<[F32, F64]>;
def UlmLamk_Shaped : AnyTypeOf<[2DTensorOf<[F32, F64]>,
                              MemRefRankOf<[F32, F64], [2]>]>;
def UlmLamk_Pivots : AnyTypeOf<[TensorRankOf<[I32], [1]>,
                              MemRefRankOf<[I32], [1]>]>;

// `init` is the capitalised operand name of the DPS init (accessor
// get<init>Mutable).
class UlmLamk_Op<string mnemonic, string init, list<Trait> traits = []>
    : Op<UlmLamk_Dialect, mnemonic, !listconcat(traits, [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<UlmLam_PropertyInferenceOpInterface>,
        DeclareOpInterfaceMethods<UlmLamk_KernelCostOpInterface>])> {
  let results = (outs Variadic<2DTensorOf<[F32, F64]>>:$result);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return get}] # init # [{Mutable();
    }
  }];
  let hasVerifier = 1;
}

// alpha/beta take the element type of the matrices.
class ScalarOf<string scalar, string shaped>
    : TypesMatchWith<scalar # " has the element type of " # shaped,
                     shaped, scalar,
                     "::llvm::cast<::mlir::ShapedType>($_self).getElementType()">;

//===----------------------------------------------------------------------===//
// Matrix level 1 (ulmBLAS geaxpy / gescal; BLAS axpy / scal on contiguous
// storage)
//===----------------------------------------------------------------------===//

def UlmLamk_GeaxpyOp : UlmLamk_Op<"geaxpy", "Y", [ScalarOf<"alpha", "X">]> {
  let summary = "Y := alpha * X + Y  (matrices)";
  let arguments = (ins UlmLamk_Float:$alpha, UlmLamk_Shaped:$X, UlmLamk_Shaped:$Y);
  let assemblyFormat = [{
    `alpha` `(` $alpha `)` `ins` `(` $X `:` type($X) `)`
    `outs` `(` $Y `:` type($Y) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GescalOp : UlmLamk_Op<"gescal", "X", [ScalarOf<"alpha", "X">]> {
  let summary = "X := alpha * X  (matrices)";
  let arguments = (ins UlmLamk_Float:$alpha, UlmLamk_Shaped:$X);
  let assemblyFormat = [{
    `alpha` `(` $alpha `)` `outs` `(` $X `:` type($X) `)`
    attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GecopyOp : UlmLamk_Op<"gecopy", "B"> {
  let summary = "B := op(A)  (a copy, transposed or not)";
  let description = [{
    A transposed operand of an in-place solve (e.g. B in M^-1 B^T, or a
    Julia array, which is the transpose of a row-major one) is copied
    transposed into a new buffer, where the solve then works.
  }];
  let arguments = (ins UlmLamk_TransAttr:$trans, UlmLamk_Shaped:$A,
                       UlmLamk_Shaped:$B);
  let assemblyFormat = [{
    `trans` `(` $trans `)` `ins` `(` $A `:` type($A) `)`
    `outs` `(` $B `:` type($B) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_AdddiagOp : UlmLamk_Op<"adddiag", "A", [ScalarOf<"alpha", "A">]> {
  let summary = "A := A + alpha * I  (A square)";
  let description = [{
    The term alpha * I of a sum (e.g. K + s2 * I in GP regression): alpha
    is added to the diagonal, the identity is never formed.  alpha may be
    a constant or a value known only at run time.
  }];
  let arguments = (ins UlmLamk_Float:$alpha, UlmLamk_Shaped:$A);
  let assemblyFormat = [{
    `alpha` `(` $alpha `)` `outs` `(` $A `:` type($A) `)`
    attr-dict (`->` type($result)^)?
  }];
}

//===----------------------------------------------------------------------===//
// BLAS 3
//===----------------------------------------------------------------------===//

def UlmLamk_GemmOp : UlmLamk_Op<"gemm", "C",
    [ScalarOf<"alpha", "A">, ScalarOf<"beta", "C">]> {
  let summary = "C := alpha * op(A) * op(B) + beta * C";
  let arguments = (ins UlmLamk_TransAttr:$trans_a, UlmLamk_TransAttr:$trans_b,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$B,
                       UlmLamk_Float:$beta, UlmLamk_Shaped:$C);
  let assemblyFormat = [{
    `trans_a` `(` $trans_a `)` `trans_b` `(` $trans_b `)`
    `alpha` `(` $alpha `)` `beta` `(` $beta `)`
    `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $C `:` type($C) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GemmtrOp : UlmLamk_Op<"gemmtr", "C",
    [ScalarOf<"alpha", "A">, ScalarOf<"beta", "C">]> {
  let summary = "C := alpha * op(A) * op(B) + beta * C, one triangle of a symmetric C";
  let description = [{
    The product is known to be symmetric (e.g. F P F^T with P symmetric):
    only the `uplo` triangle of C is computed (half the flops of gemm) and
    read; consumers that need the full matrix get an explicit
    symmetrisation inserted by kernel selection, as for syrk.
  }];
  let arguments = (ins UlmLamk_UploAttr:$uplo, UlmLamk_TransAttr:$trans_a,
                       UlmLamk_TransAttr:$trans_b,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$B,
                       UlmLamk_Float:$beta, UlmLamk_Shaped:$C);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)` `trans_a` `(` $trans_a `)` `trans_b` `(` $trans_b `)`
    `alpha` `(` $alpha `)` `beta` `(` $beta `)`
    `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $C `:` type($C) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_SymmOp : UlmLamk_Op<"symm", "C",
    [ScalarOf<"alpha", "A">, ScalarOf<"beta", "C">]> {
  let summary = "C := alpha A op(B) + beta C (left) or alpha op(B) A + beta C (right), A symmetric";
  let description = [{
    Only the `uplo` triangle of A is read (a symmetric matrix stored as
    one triangle).  B is general, op(B) = B or B^T (`trans_b`).
  }];
  let arguments = (ins UlmLamk_SideAttr:$side, UlmLamk_UploAttr:$uplo,
                       UlmLamk_TransAttr:$trans_b,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$B,
                       UlmLamk_Float:$beta, UlmLamk_Shaped:$C);
  let assemblyFormat = [{
    `side` `(` $side `)` `uplo` `(` $uplo `)` `trans_b` `(` $trans_b `)`
    `alpha` `(` $alpha `)` `beta` `(` $beta `)`
    `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $C `:` type($C) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_TrmmOp : UlmLamk_Op<"trmm", "B", [ScalarOf<"alpha", "A">]> {
  let summary = "B := alpha * op(A) * B  (A triangular, in place)";
  let arguments = (ins UlmLamk_SideAttr:$side, UlmLamk_UploAttr:$uplo,
                       UlmLamk_TransAttr:$trans, UlmLamk_DiagAttr:$diag,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$B);
  let assemblyFormat = [{
    `side` `(` $side `)` `uplo` `(` $uplo `)` `trans` `(` $trans `)`
    `diag` `(` $diag `)` `alpha` `(` $alpha `)`
    `ins` `(` $A `:` type($A) `)` `outs` `(` $B `:` type($B) `)`
    attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_TrsmOp : UlmLamk_Op<"trsm", "B", [ScalarOf<"alpha", "A">]> {
  let summary = "B := alpha * op(A)^-1 * B  (A triangular, in place)";
  let arguments = (ins UlmLamk_SideAttr:$side, UlmLamk_UploAttr:$uplo,
                       UlmLamk_TransAttr:$trans, UlmLamk_DiagAttr:$diag,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$B);
  let assemblyFormat = [{
    `side` `(` $side `)` `uplo` `(` $uplo `)` `trans` `(` $trans `)`
    `diag` `(` $diag `)` `alpha` `(` $alpha `)`
    `ins` `(` $A `:` type($A) `)` `outs` `(` $B `:` type($B) `)`
    attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_SyrkOp : UlmLamk_Op<"syrk", "C",
    [ScalarOf<"alpha", "A">, ScalarOf<"beta", "C">]> {
  let summary = "C := alpha * op(A) * op(A)^T + beta * C (one triangle)";
  let description = [{
    Only the `uplo` triangle of C is written.  The result type carries
    `symmetric`; consumers that need the full matrix get an explicit
    symmetrisation inserted by kernel selection.
  }];
  let arguments = (ins UlmLamk_UploAttr:$uplo, UlmLamk_TransAttr:$trans,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A,
                       UlmLamk_Float:$beta, UlmLamk_Shaped:$C);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)` `trans` `(` $trans `)`
    `alpha` `(` $alpha `)` `beta` `(` $beta `)`
    `ins` `(` $A `:` type($A) `)` `outs` `(` $C `:` type($C) `)`
    attr-dict (`->` type($result)^)?
  }];
}

//===----------------------------------------------------------------------===//
// BLAS 2
//===----------------------------------------------------------------------===//

def UlmLamk_SymvOp : UlmLamk_Op<"symv", "Y",
    [ScalarOf<"alpha", "A">, ScalarOf<"beta", "y">]> {
  let summary = "y := alpha * A * x + beta * y (A symmetric, uplo triangle)";
  let arguments = (ins UlmLamk_UploAttr:$uplo,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$x,
                       UlmLamk_Float:$beta, UlmLamk_Shaped:$y);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)` `alpha` `(` $alpha `)` `beta` `(` $beta `)`
    `ins` `(` $A `,` $x `:` type($A) `,` type($x) `)`
    `outs` `(` $y `:` type($y) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_SycompleteOp : UlmLamk_Op<"sycomplete", "A"> {
  let summary = "complete a symmetric matrix: mirror the uplo triangle";
  let description = [{
    A symmetric matrix of which only the `uplo` triangle is stored becomes
    a full (general) matrix: the other triangle is overwritten with the
    mirror image.  In place, as every DPS op: bufferization works on the
    matrix itself when it may be overwritten (a temporary, or an inout
    parameter), and on a copy otherwise.  n^2 / 2 element copies.
  }];
  let arguments = (ins UlmLamk_UploAttr:$uplo, UlmLamk_Shaped:$A);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)` `outs` `(` $A `:` type($A) `)`
    attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GemvOp : UlmLamk_Op<"gemv", "Y",
    [ScalarOf<"alpha", "A">, ScalarOf<"beta", "y">]> {
  let summary = "y := alpha * op(A) * x + beta * y";
  let arguments = (ins UlmLamk_TransAttr:$trans,
                       UlmLamk_Float:$alpha, UlmLamk_Shaped:$A, UlmLamk_Shaped:$x,
                       UlmLamk_Float:$beta, UlmLamk_Shaped:$y);
  let assemblyFormat = [{
    `trans` `(` $trans `)` `alpha` `(` $alpha `)` `beta` `(` $beta `)`
    `ins` `(` $A `,` $x `:` type($A) `,` type($x) `)`
    `outs` `(` $y `:` type($y) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_TrsvOp : UlmLamk_Op<"trsv", "X"> {
  let summary = "x := op(A)^-1 * x  (A triangular, in place)";
  let arguments = (ins UlmLamk_UploAttr:$uplo, UlmLamk_TransAttr:$trans,
                       UlmLamk_DiagAttr:$diag,
                       UlmLamk_Shaped:$A, UlmLamk_Shaped:$x);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)` `trans` `(` $trans `)` `diag` `(` $diag `)`
    `ins` `(` $A `:` type($A) `)` `outs` `(` $x `:` type($x) `)`
    attr-dict (`->` type($result)^)?
  }];
}

//===----------------------------------------------------------------------===//
// LAPACK
//===----------------------------------------------------------------------===//

def UlmLamk_PotrfOp : UlmLamk_Op<"potrf", "A"> {
  let summary = "Cholesky factorisation A = L L^T (in place)";
  let description = [{
    Writes the factor into the `uplo` triangle; the other triangle is left
    untouched and must not be read as part of L.  `info` is LAPACK's: 0, or
    i > 0 if the leading minor of order i is not positive definite (the
    `spd` assumption did not hold numerically).  ulm-lam-info turns it into
    the info of the function.
  }];
  let arguments = (ins UlmLamk_UploAttr:$uplo, UlmLamk_Shaped:$A);
  let results = (outs Variadic<2DTensorOf<[F32, F64]>>:$result, I64:$info);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)`
    `outs` `(` $A `:` type($A) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GetrfOp : Op<UlmLamk_Dialect, "getrf", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<UlmLamk_KernelCostOpInterface>]> {
  let summary = "LU factorisation with partial pivoting P A = L U (in place)";
  let description = [{
    `info` is LAPACK's: 0, or i > 0 if U(i-1, i-1) is exactly zero (the
    matrix is singular).  With `symmetric`: A symmetric (not positive
    definite), A = L D L^T by Bunch-Kaufman pivoting (LAPACK's sytrf), half
    the flops; getrs with `symmetric` solves with it (sytrs).
  }];
  let arguments = (ins UlmLamk_Shaped:$A, UlmLamk_Pivots:$ipiv,
                       UnitAttr:$symmetric);
  let results = (outs Variadic<AnyRankedTensor>:$result, I64:$info);
  let assemblyFormat = [{
    `outs` `(` $A `,` $ipiv `:` type($A) `,` type($ipiv) `)`
    attr-dict (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 0, 2);
    }
  }];
}

def UlmLamk_GelsOp : Op<UlmLamk_Dialect, "gels", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<UlmLamk_KernelCostOpInterface>]> {
  let summary = "least squares by QR: B := the X minimising ||A X - B|| in its leading rows (in place)";
  let description = [{
    A (m x n, m >= n) is overwritten by its QR factorisation, B (m x k) by
    the solution in its leading n rows (LAPACK's gels).  `info` is LAPACK's:
    0, or i > 0 if R(i-1, i-1) is exactly zero (A has no full column rank).
  }];
  let arguments = (ins UlmLamk_Shaped:$A, UlmLamk_Shaped:$B);
  let results = (outs Variadic<AnyRankedTensor>:$result, I64:$info);
  let assemblyFormat = [{
    `outs` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    attr-dict (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 0, 2);
    }
  }];
}

def UlmLamk_PotriOp : UlmLamk_Op<"potri", "A"> {
  let summary = "A^-1 from the Cholesky factor of potrf (in place)";
  let description = [{
    A holds the factor of potrf in its `uplo` triangle; afterwards that
    triangle holds A^-1 (symmetric; the other triangle is untouched).  An
    explicit inverse where solves are not enough (an inverse used as a
    value): n^3 / 3 flops besides potrf's n^3 / 3, a third of solving with
    the identity.
  }];
  let arguments = (ins UlmLamk_UploAttr:$uplo, UlmLamk_Shaped:$A);
  let assemblyFormat = [{
    `uplo` `(` $uplo `)` `outs` `(` $A `:` type($A) `)`
    attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GetriOp : UlmLamk_Op<"getri", "A"> {
  let summary = "A^-1 from the LU factors of getrf (in place)";
  let description = [{
    A holds the factors of getrf, ipiv its pivots; afterwards A holds A^-1.
    4 n^3 / 3 flops besides getrf's 2 n^3 / 3 (solving with the identity:
    2 n^3).
  }];
  let arguments = (ins UlmLamk_Pivots:$ipiv, UlmLamk_Shaped:$A);
  let assemblyFormat = [{
    `ins` `(` $ipiv `:` type($ipiv) `)`
    `outs` `(` $A `:` type($A) `)` attr-dict (`->` type($result)^)?
  }];
}

def UlmLamk_GetrsOp : UlmLamk_Op<"getrs", "B"> {
  let summary = "B := op(A)^-1 B (left) or B op(A)^-1 (right), LU factors";
  let description = [{
    side(right) is the solve from the right, B op(A)^-1 = (op(A)^-T B^T)^T:
    a solve from the left on the transposed view of the same buffer (the
    lowering swaps the strides of B and flips trans), in place.
  }];
  let arguments = (ins
      DefaultValuedAttr<UlmLamk_SideAttr, "::mlir::ulm_lamk::Side::left">:$side,
      UlmLamk_TransAttr:$trans, UlmLamk_Shaped:$LU, UlmLamk_Pivots:$ipiv,
      UlmLamk_Shaped:$B, UnitAttr:$symmetric);
  let assemblyFormat = [{
    (`side` `(` $side^ `)`)? `trans` `(` $trans `)`
    `ins` `(` $LU `,` $ipiv `:` type($LU) `,` type($ipiv) `)`
    `outs` `(` $B `:` type($B) `)` attr-dict (`->` type($result)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Calls of external functions that update matrices in place
//===----------------------------------------------------------------------===//

def UlmLamk_CallOp : Op<UlmLamk_Dialect, "call", [
    DestinationStyleOpInterface, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "call an external function that updates matrices in place";
  let description = [{
    `%R1 = ulm_lamk.call @update_covariance(%i) outs(%R0 : t) -> t` calls the
    external function with the inputs followed by the buffers of the
    `outs` operands, which it updates in place (the ulm_lam ABI: matrices by
    reference, the caller owns them).  In destination-passing style, so
    One-Shot Bufferize keeps a matrix that is updated in every step of a
    loop in one buffer.  The callee is unknown: the op reads and writes
    memory, so it is never hoisted, duplicated or removed.

    Created by ulm-lam-results-to-out-params from a `func.call` of a
    declaration whose result is marked `{ulm_lam.inout = k}`; the in-place
    arguments must be the trailing arguments of the function.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
                       Variadic<UlmLamk_Shaped>:$inits);
  let results = (outs Variadic<2DTensorOf<[F32, F64]>>:$result);
  let assemblyFormat = [{
    $callee `(` $inputs `)` `outs` `(` $inits `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getInitsMutable();
    }
  }];
}

#endif // ULM_LAM_KERNEL_OPS_TD
