// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, loop-invariant-code-motion, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' \
// RUN:   | ulm-lam-opt --pass-pipeline='builtin.module(ulm-lam-results-to-out-params, eliminate-empty-tensors,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-results-to-out-params{hoist-static-allocs modify-public-functions add-result-attr},func.func(buffer-loop-hoisting), buffer-deallocation-pipeline, canonicalize,ulm-lam-workspace, canonicalize, cse, ulm-lam-info, canonicalize)' \
// RUN:   | ulm-lam-opt --pass-pipeline='builtin.module(convert-ulm-lamk-to-ulmblas, reconcile-unrealized-casts)' \
// RUN:   | FileCheck %s --check-prefix=ULMBLAS

// LAPACK style (examples/calls): bar_ overwrites y with the solution, foo_
// calls it twice on the same y, so neither has a workspace; foo_matlab
// copies y directly into its result, M and N into its workspace, and lets
// foo_ work there.
//
// bar_ and foo_: the matrices and y by reference, nothing else; the calls
// pass foo_'s y on, without a copy.
// ULMBLAS-LABEL: emitc.func @bar_(%arg0: !emitc.size_t, %arg1: !emitc.ptr<f64> {{{.*}}}, %arg2: !emitc.ptr<f64> {{{.*}}}) -> i64
// ULMBLAS-NOT: ulm_lam_copy
// ULMBLAS-LABEL: emitc.func @foo_(%arg0: !emitc.size_t, %arg1: !emitc.ptr<f64> {{{.*}}}, %arg2: !emitc.ptr<f64> {{{.*}}}, %arg3: !emitc.ptr<f64> {{{.*}}}) -> i64
// ULMBLAS-NOT: ulm_lam_copy
// ULMBLAS: call @bar_(%{{.*}}, %{{.*}}, %arg3)
// ULMBLAS-NOT: ulm_lam_copy
// ULMBLAS: call @bar_(%{{.*}}, %{{.*}}, %arg3)
// foo_matlab: y copied into the result x, then foo_ computes in x; no copy
// after the call.
// ULMBLAS-LABEL: emitc.func @foo_matlab(
// ULMBLAS: call_opaque "ulm_lam_copy"(%arg4, %arg3,
// ULMBLAS: call @foo_(%{{.*}}, %{{.*}}, %{{.*}}, %arg4)
// ULMBLAS-NOT: ulm_lam_copy
// ULMBLAS: return

module {
  func.func @bar_(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.inout}, %arg2: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout = 2 : i64}) {
    %0 = ulm_lam.inverse %arg1 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
    %1 = ulm_lam.mul %0, %arg2 : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
  func.func @foo_(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.inout}, %arg2: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.inout}, %arg3: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout = 3 : i64}) {
    %0 = call @bar_(%arg0, %arg2, %arg3) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    %1 = call @bar_(%arg0, %arg1, %0) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
  func.func @foo_matlab(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg2: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg3: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.out}) {
    %0 = call @foo_(%arg0, %arg1, %arg2, %arg3) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    return %0 : tensor<?x1xf64>
  }
}

