// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, loop-invariant-code-motion, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' \
// RUN:   | ulm-lam-opt --pass-pipeline='builtin.module(ulm-lam-results-to-out-params, eliminate-empty-tensors,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-results-to-out-params{hoist-static-allocs modify-public-functions add-result-attr},func.func(buffer-loop-hoisting), buffer-deallocation-pipeline, canonicalize,ulm-lam-workspace, canonicalize, cse, ulm-lam-info, canonicalize)' \
// RUN:   | ulm-lam-opt --pass-pipeline='builtin.module(convert-ulm-lamk-to-ulmblas, reconcile-unrealized-casts)' \
// RUN:   | FileCheck %s --check-prefix=ULMBLAS
// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, loop-invariant-code-motion, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' \
// RUN:   | ulm-lam-opt --pass-pipeline='builtin.module(ulm-lam-results-to-out-params, eliminate-empty-tensors,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-results-to-out-params{hoist-static-allocs modify-public-functions add-result-attr},func.func(buffer-loop-hoisting), buffer-deallocation-pipeline, canonicalize,ulm-lam-workspace, canonicalize, cse, ulm-lam-info, canonicalize)' \
// RUN:   | ulm-lam-opt --pass-pipeline='builtin.module(convert-ulm-lamk-to-ulmblas{backend=cblas}, reconcile-unrealized-casts)' \
// RUN:   | FileCheck %s --check-prefix=CBLAS

// The whole pipeline for L B, L lower triangular, down to the calls: the
// copy of B into the result, then trlmm in place; on ulmBLAS and on CBLAS.

// ULMBLAS-LABEL: @trmm
// ULMBLAS: call_opaque "ulm_lam_gecopy"
// ULMBLAS: call_opaque "ulmBLAS::trlmm
// ULMBLAS-NOT: gemm
// CBLAS-LABEL: @trmm
// CBLAS: call_opaque "ulm_lam_gecopy"
// CBLAS: call_opaque "ulm_lam_cblas::trlmm
func.func @trmm(%n: index, %m: index, %l: tensor<?x?xf64, #ulm_lam.props<lower_tri>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %b: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]}) -> (tensor<?x?xf64> {ulm_lam.out, ulm_lam.dims = [0, 1]}) {
  %p = ulm_lam.mul %l, %b : tensor<?x?xf64, #ulm_lam.props<lower_tri>>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %p : tensor<?x?xf64>
}
