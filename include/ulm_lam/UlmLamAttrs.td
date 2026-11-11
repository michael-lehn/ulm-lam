//===- UlmLamAttrs.td - matrix properties and kernel flags --*- tablegen -*-===//

#ifndef ULM_LAM_ATTRS_TD
#define ULM_LAM_ATTRS_TD

include "ulm_lam/UlmLamBase.td"
include "mlir/IR/TensorEncoding.td"

//===----------------------------------------------------------------------===//
// Matrix properties: #ulm_lam.props<lower_tri, non_singular>
//
// A set of facts about a matrix value.  The set is always interpreted
// modulo the implication closure implemented in C++ (`ulm_lam::closure`):
//
//   spd          => spsd, symmetric, non_singular, square
//   spsd         => symmetric, square
//   symmetric    => square
//   identity     => diagonal, spd, orthogonal, permutation, unit_diag
//   permutation  => orthogonal
//   orthogonal   => full_rank            (orthonormal columns)
//   square & full_rank <=> non_singular
//   lower_tri & upper_tri  ==  diagonal  (a case group, not a bit)
//
// The encoding is an *assertion*: absence of a bit means "unknown", never
// "false".  Dropping bits is always sound.
//===----------------------------------------------------------------------===//

def UlmLam_PropNone       : I32BitEnumAttrCaseNone<"general">;
def UlmLam_PropSquare     : I32BitEnumAttrCaseBit<"square",       0>;
def UlmLam_PropSymmetric  : I32BitEnumAttrCaseBit<"symmetric",    1>;
def UlmLam_PropSPSD       : I32BitEnumAttrCaseBit<"spsd",         2>;
def UlmLam_PropSPD        : I32BitEnumAttrCaseBit<"spd",          3>;
def UlmLam_PropLowerTri   : I32BitEnumAttrCaseBit<"lower_tri",    4>;
def UlmLam_PropUpperTri   : I32BitEnumAttrCaseBit<"upper_tri",    5>;
def UlmLam_PropUnitDiag   : I32BitEnumAttrCaseBit<"unit_diag",    6>;
def UlmLam_PropIdentity   : I32BitEnumAttrCaseBit<"identity",     7>;
def UlmLam_PropZero       : I32BitEnumAttrCaseBit<"zero",         8>;
def UlmLam_PropOrthogonal : I32BitEnumAttrCaseBit<"orthogonal",   9>;
def UlmLam_PropPermutation: I32BitEnumAttrCaseBit<"permutation", 10>;
def UlmLam_PropNonSingular: I32BitEnumAttrCaseBit<"non_singular",11>;
def UlmLam_PropFullRank   : I32BitEnumAttrCaseBit<"full_rank",   12>;
def UlmLam_PropDiagonal   : I32BitEnumAttrCaseGroup<"diagonal",
                              [UlmLam_PropLowerTri, UlmLam_PropUpperTri]>;

def UlmLam_Properties : I32BitEnumAttr<"Properties", "matrix properties", [
    UlmLam_PropNone, UlmLam_PropSquare, UlmLam_PropSymmetric, UlmLam_PropSPSD,
    UlmLam_PropSPD, UlmLam_PropLowerTri, UlmLam_PropUpperTri, UlmLam_PropUnitDiag,
    UlmLam_PropIdentity, UlmLam_PropZero, UlmLam_PropOrthogonal,
    UlmLam_PropPermutation, UlmLam_PropNonSingular, UlmLam_PropFullRank,
    UlmLam_PropDiagonal]> {
  let separator = ", ";
  let cppNamespace = "::mlir::ulm_lam";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

// Used as a builtin tensor encoding:  tensor<?x?xf64, #ulm_lam.props<spd>>
def UlmLam_PropsAttr : EnumAttr<UlmLam_Dialect, UlmLam_Properties, "props",
    [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let assemblyFormat = "`<` $value `>`";
  let description = [{
    Matrix properties attached to a rank-2 tensor.  `verifyEncoding` rejects
    contradictions (e.g. `zero` with `non_singular`) and properties that
    contradict a static shape (e.g. `square` on a 3x4 tensor).
  }];
}

//===----------------------------------------------------------------------===//
// Kernel flags (BLAS conventions)
//===----------------------------------------------------------------------===//

def UlmLamk_Side : I32EnumAttr<"Side", "side of the triangular operand", [
    I32EnumAttrCase<"left", 0>, I32EnumAttrCase<"right", 1>]> {
  let cppNamespace = "::mlir::ulm_lamk";
  let genSpecializedAttr = 0;
}
def UlmLamk_SideAttr : EnumAttr<UlmLamk_Dialect, UlmLamk_Side, "side"> {
  let assemblyFormat = "$value";
}

def UlmLamk_Uplo : I32EnumAttr<"Uplo", "referenced triangle", [
    I32EnumAttrCase<"lower", 0>, I32EnumAttrCase<"upper", 1>]> {
  let cppNamespace = "::mlir::ulm_lamk";
  let genSpecializedAttr = 0;
}
def UlmLamk_UploAttr : EnumAttr<UlmLamk_Dialect, UlmLamk_Uplo, "uplo"> {
  let assemblyFormat = "$value";
}

def UlmLamk_Trans : I32EnumAttr<"Trans", "operand transposition", [
    I32EnumAttrCase<"none", 0>, I32EnumAttrCase<"trans", 1>]> {
  let cppNamespace = "::mlir::ulm_lamk";
  let genSpecializedAttr = 0;
}
def UlmLamk_TransAttr : EnumAttr<UlmLamk_Dialect, UlmLamk_Trans, "trans"> {
  let assemblyFormat = "$value";
}

def UlmLamk_Diag : I32EnumAttr<"Diag", "unit diagonal", [
    I32EnumAttrCase<"non_unit", 0>, I32EnumAttrCase<"unit", 1>]> {
  let cppNamespace = "::mlir::ulm_lamk";
  let genSpecializedAttr = 0;
}
def UlmLamk_DiagAttr : EnumAttr<UlmLamk_Dialect, UlmLamk_Diag, "diag"> {
  let assemblyFormat = "$value";
}

#endif // ULM_LAM_ATTRS_TD
