// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s

// A S A^T with S spd: S = L L^T, W = A L (trmm), W W^T (syrk) when
// n + 2400 < 3 m (S n x n, A m x n); otherwise gemm and gemmtr; with
// sizes known at run time only, both behind an scf.if.

// CHECK-LABEL: func.func @large
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.trmm side(right) uplo(lower)
// CHECK: ulm_lamk.syrk
func.func @large(%f: tensor<3000x3000xf64>, %p: tensor<3000x3000xf64, #ulm_lam.props<spd>>) -> tensor<3000x3000xf64> {
  %ft = ulm_lam.transpose %f : tensor<3000x3000xf64> -> tensor<3000x3000xf64>
  %m = ulm_lam.mul %f, %p, %ft : tensor<3000x3000xf64>, tensor<3000x3000xf64, #ulm_lam.props<spd>>, tensor<3000x3000xf64> -> tensor<3000x3000xf64>
  return %m : tensor<3000x3000xf64>
}

// CHECK-LABEL: func.func @small
// CHECK-NOT: ulm_lamk.potrf
// CHECK: ulm_lamk.gemmtr
func.func @small(%f: tensor<500x500xf64>, %p: tensor<500x500xf64, #ulm_lam.props<spd>>) -> tensor<500x500xf64> {
  %ft = ulm_lam.transpose %f : tensor<500x500xf64> -> tensor<500x500xf64>
  %m = ulm_lam.mul %f, %p, %ft : tensor<500x500xf64>, tensor<500x500xf64, #ulm_lam.props<spd>>, tensor<500x500xf64> -> tensor<500x500xf64>
  return %m : tensor<500x500xf64>
}

// CHECK-LABEL: func.func @dynamic
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.trmm
// CHECK: ulm_lamk.syrk
// CHECK: } else {
// CHECK: ulm_lamk.gemm
// CHECK: ulm_lamk.gemmtr
func.func @dynamic(%n: index, %f: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %q: tensor<?x?xf64, #ulm_lam.props<spsd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}) -> (tensor<?x?xf64> {ulm_lam.out, ulm_lam.dims = [0, 0]}) {
  %ft = ulm_lam.transpose %f : tensor<?x?xf64> -> tensor<?x?xf64>
  %m = ulm_lam.mul %f, %p, %ft : tensor<?x?xf64>, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x?xf64> -> tensor<?x?xf64>
  %s = ulm_lam.add %m, %q : tensor<?x?xf64>, tensor<?x?xf64, #ulm_lam.props<spsd>> -> tensor<?x?xf64>
  return %s : tensor<?x?xf64>
}
