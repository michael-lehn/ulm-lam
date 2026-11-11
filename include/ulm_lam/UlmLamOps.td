//===- UlmLamOps.td - ulm_lam dialect operations -----------*- tablegen -*-===//
//
// All ulm_lam ops are pure and operate on rank-2 tensors (vectors are n x 1,
// row vectors 1 x n, scalars are plain floats).  Result types are inferred
// (InferTypeOpInterface) but *compatibility* is relaxed: a result type is
// accepted if its shape matches and its properties are implied by the
// inferred ones.  This makes property refinement optional and keeps the IR
// valid while rewrite patterns replace operands by better-typed values.
//
//===----------------------------------------------------------------------===//

#ifndef ULM_LAM_OPS_TD
#define ULM_LAM_OPS_TD

include "ulm_lam/UlmLamBase.td"
include "ulm_lam/UlmLamAttrs.td"
include "ulm_lam/UlmLamInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def UlmLam_Matrix : 2DTensorOf<[F32, F64]>;

class UlmLam_Op<string mnemonic, list<Trait> traits = []>
    : Op<UlmLam_Dialect, mnemonic, !listconcat(traits, [
        Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        DeclareOpInterfaceMethods<UlmLam_PropertyInferenceOpInterface>])> {
  code extraUlmLamDeclaration = [{
    /// Shapes equal, and the properties of `actual` are implied by the
    /// properties of `inferred`.
    static bool isCompatibleReturnTypes(::mlir::TypeRange inferred,
                                        ::mlir::TypeRange actual);
  }];
  let extraClassDeclaration = extraUlmLamDeclaration;
}

//===----------------------------------------------------------------------===//
// Algebra
//===----------------------------------------------------------------------===//

def UlmLam_MulOp : UlmLam_Op<"mul"> {
  let summary = "n-ary matrix product (associative, not commutative)";
  let description = [{
    `ulm_lam.mul %A, %B, %C` is the product A*B*C without a parenthesisation.
    Choosing the parenthesisation is part of kernel selection (generalised
    matrix chain problem).  Nested products are flattened by the
    canonicalizer; the verifier checks inner dimensions.

    ```mlir
    %S = ulm_lam.mul %Xt, %Mi, %X
       : tensor<200x2000xf64>, tensor<2000x2000xf64, #spd>,
         tensor<2000x200xf64, #fr> -> tensor<200x200xf64, #spd>
    ```
  }];
  let arguments = (ins Variadic<UlmLam_Matrix>:$factors);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat =
      "$factors attr-dict `:` type($factors) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def UlmLam_AddOp : UlmLam_Op<"add", [Commutative]> {
  let summary = "n-ary matrix sum (associative and commutative)";
  let arguments = (ins Variadic<UlmLam_Matrix>:$terms);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat =
      "$terms attr-dict `:` type($terms) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def UlmLam_ScaleOp : UlmLam_Op<"scale"> {
  let summary = "scalar times matrix";
  let description = [{
    Subtraction is `add(A, scale(-1, B))`.  The canonicalizer hoists scales
    out of products so that they end up in the alpha/beta of a kernel.
  }];
  let arguments = (ins AnyFloat:$alpha, UlmLam_Matrix:$operand);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat = [{
    $alpha `,` $operand attr-dict `:` type($alpha) `,` type($operand)
    `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def UlmLam_TransposeOp : UlmLam_Op<"transpose"> {
  let summary = "matrix transpose";
  let description = [{
    Normal form pushes transposes to the leaves:
    (AB)^T -> B^T A^T,  (A+B)^T -> A^T + B^T,  (A^T)^T -> A,
    A^T -> A if A is symmetric.
  }];
  let arguments = (ins UlmLam_Matrix:$operand);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasCanonicalizer = 1;
}

def UlmLam_InverseOp : UlmLam_Op<"inverse"> {
  let summary = "matrix inverse (operand must be square)";
  let description = [{
    Never computed explicitly unless the inverse itself is requested.
    Kernel selection maps products with inverses onto solves
    (trsm/trsv/getrs/potrs) after factorising the operand.
    Normal form: (AB)^-1 -> B^-1 A^-1 for square factors,
    (A^T)^-1 -> (A^-1)^T, i.e. transpose is the outermost leaf decoration.
  }];
  let arguments = (ins UlmLam_Matrix:$operand);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Constants and assertions
//===----------------------------------------------------------------------===//

def UlmLam_IdentityOp : Op<UlmLam_Dialect, "identity", [Pure,
    DeclareOpInterfaceMethods<UlmLam_PropertyInferenceOpInterface>]> {
  let summary = "identity matrix";
  let description = [{
    The result type is explicit (its sizes cannot be inferred from the
    operands); the result always carries `identity`.
  }];
  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat = "(`(` $dynamicSizes^ `)`)? attr-dict `:` type($result)";
}

def UlmLam_AssumeOp : Op<UlmLam_Dialect, "assume", [Pure,
    DeclareOpInterfaceMethods<UlmLam_PropertyInferenceOpInterface>]> {
  let summary = "change the asserted properties of a value";
  let description = [{
    The result is the operand with the properties of the result type.
    Properties the operand does not have are *assertions*: with `checked`
    they are verified at run time (e.g. symmetry test, or success of the
    Cholesky factorisation for spd), otherwise trusted.  This is the only
    op that may add facts that inference cannot derive.  Dropping
    properties is always sound; rewrite patterns use that to keep result
    types unchanged when a rewrite infers stronger properties.
  }];
  let arguments = (ins UlmLam_Matrix:$operand, UnitAttr:$checked);
  let results = (outs UlmLam_Matrix:$result);
  let assemblyFormat = [{
    $operand (`checked` $checked^)? attr-dict `:` type($operand)
    `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // ULM_LAM_OPS_TD
