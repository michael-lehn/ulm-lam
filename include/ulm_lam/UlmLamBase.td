//===- UlmLamBase.td - ulm_lam and ulm_lamk dialect definitions --*- tablegen -*-===//
//
// Two dialects:
//
//   ulm_lam   -- linear algebra at the level of the mathematical problem:
//             value-semantic, n-ary products and sums, transpose, inverse,
//             matrix properties carried in the tensor encoding.
//
//   ulm_lamk  -- linear algebra *kernels*: one op per BLAS/LAPACK routine,
//             destination-passing style, works on tensors and memrefs.
//
// Kernel selection (the actual LAMP) is the conversion ulm_lam -> ulm_lamk.
//
//===----------------------------------------------------------------------===//

#ifndef ULM_LAM_BASE_TD
#define ULM_LAM_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def UlmLam_Dialect : Dialect {
  let name = "ulm_lam";
  let cppNamespace = "::mlir::ulm_lam";
  let summary = "Linear algebra expressions with matrix properties";
  let description = [{
    The `ulm_lam` dialect describes a linear algebra problem the way it is
    written on paper: `b = (X^T M^-1 X)^-1 X^T M^-1 y`.  Operations are pure
    and operate on rank-2 builtin tensors.  Matrix properties (SPD, lower
    triangular, full rank, ...) are carried as `#ulm_lam.props<...>` tensor
    encodings and are inferred for every result.
  }];
  let useDefaultAttributePrinterParser = 1;
  let hasCanonicalizer = 1;
  let dependentDialects = ["::mlir::tensor::TensorDialect",
                           "::mlir::arith::ArithDialect"];
}

def UlmLamk_Dialect : Dialect {
  let name = "ulm_lamk";
  let cppNamespace = "::mlir::ulm_lamk";
  let summary = "BLAS/LAPACK kernels in destination-passing style";
  let description = [{
    One operation per kernel (gemm, trsm, syrk, potrf, ...).  Operations
    follow destination-passing style so that One-Shot Bufferize decides
    in-place updates and copies.  Each op implements `KernelCostOpInterface`.
  }];
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["::mlir::ulm_lam::UlmLamDialect"];
}

#endif // ULM_LAM_BASE_TD
