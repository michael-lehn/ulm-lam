//===- UlmLamInterfaces.td - property inference and cost ---*- tablegen -*-===//

#ifndef ULM_LAM_INTERFACES_TD
#define ULM_LAM_INTERFACES_TD

include "mlir/IR/OpBase.td"

// Implemented by every ulm_lam op *and* every ulm_lamk op.  Kernel selection
// needs the properties of kernel results (e.g. W = L^-1 X is full_rank if
// X is) to keep inferring properties after part of the problem is mapped.
def UlmLam_PropertyInferenceOpInterface
    : OpInterface<"PropertyInferenceOpInterface"> {
  let cppNamespace = "::mlir::ulm_lam";
  let description = [{
    Computes properties of the op's results from the properties of its
    operands.  Operand *identity* matters, not only operand properties:
    `mul(transpose(%A), %A)` is spsd for any %A, and spd if %A is full rank.
    The default implementation reads the operands' tensor encodings.
  }];
  let methods = [
    InterfaceMethod<
      "Infer result properties from operand properties.",
      "::mlir::ulm_lam::Properties", "inferResultProperties",
      (ins "unsigned":$resultIdx,
           "::llvm::ArrayRef<::mlir::ulm_lam::Properties>":$operandProps)>
  ];
}

// Implemented by ulm_lamk ops.  The cost model of kernel selection.
def UlmLamk_KernelCostOpInterface : OpInterface<"KernelCostOpInterface"> {
  let cppNamespace = "::mlir::ulm_lamk";
  let description = [{
    Estimated cost of executing the kernel.  Sizes come from static shapes
    or, for dynamic dimensions, from `ulm_lam.size_hint` on function arguments.
    The first cost model is FLOPs (as in Linnea); a second one can weight
    FLOPs by measured per-kernel efficiency.
  }];
  let methods = [
    InterfaceMethod<
      "Number of floating point operations.",
      "::mlir::FailureOr<double>", "estimateFlops",
      (ins "const ::mlir::ulm_lam::SizeAssignment &":$sizes)>,
    InterfaceMethod<
      "Bytes moved (for memory-bound kernels such as gemv/trsv).",
      "::mlir::FailureOr<double>", "estimateBytes",
      (ins "const ::mlir::ulm_lam::SizeAssignment &":$sizes)>
  ];
}

#endif // ULM_LAM_INTERFACES_TD
