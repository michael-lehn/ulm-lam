// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, loop-invariant-code-motion, ulm-lam-select-kernels))' | FileCheck %s

// The linear layer of the notebook as the Julia frontend traces it: every
// matrix from Julia is the transpose of the row-major value.  The loop
// carries W in Julia's layout (FlipTransposedLoopState), and the residual
// R = X W + 1 b^T - Y is computed transposed, in the buffer of Y (selectAdd):
// four BLAS calls per step, no copy, every transpose a flag.

// CHECK-LABEL: func.func @train
// CHECK-NOT: gecopy
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg3,
// CHECK: %[[Y:.*]] = func.call @batch_y
// CHECK-NOT: gecopy
// CHECK: ulm_lamk.gemm {{.*}} outs(%[[Y]] :
// CHECK-NOT: gecopy
// CHECK: ulm_lamk.gemm
// CHECK-NOT: gecopy
// CHECK: ulm_lamk.gemv
// CHECK-NOT: gecopy
// CHECK: ulm_lamk.gemm
// CHECK-NOT: gecopy
// CHECK: scf.yield
// CHECK-NOT: gecopy
// CHECK: return
func.func private @batch_y(index, index, index) -> (tensor<?x?xf64> {ulm_lam.dims = [1, 0]})

func.func private @batch_x(index, index, index) -> (tensor<?x?xf64> {ulm_lam.dims = [1, 0]})

func.func @train(%s.p: index, %s.k: index, %s.B: index, %p.W0: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [1, 0]}, %p.b0: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [1]}, %p.ones: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [2]}, %p.step: f64, %p.steps: index) -> (tensor<?x?xf64> {ulm_lam.out, ulm_lam.dims = [1, 0]}, tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [1]}) {
  %v1 = ulm_lam.transpose %p.W0 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v2 = arith.constant 0 : index
  %v3 = arith.constant 1 : index
  %v4:2 = scf.for %v5 = %v2 to %p.steps step %v3 iter_args(%v6 = %v1, %v7 = %p.b0) -> (tensor<?x?xf64>, tensor<?x1xf64>) {
    %v8 = func.call @batch_x(%s.B, %s.p, %v5) : (index, index, index) -> tensor<?x?xf64>
    %v9 = ulm_lam.transpose %v8 : tensor<?x?xf64> -> tensor<?x?xf64>
    %v10 = func.call @batch_y(%s.B, %s.k, %v5) : (index, index, index) -> tensor<?x?xf64>
    %v11 = ulm_lam.transpose %v10 : tensor<?x?xf64> -> tensor<?x?xf64>
    %v12 = ulm_lam.mul %v9, %v6 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
    %v13 = ulm_lam.transpose %v7 : tensor<?x1xf64> -> tensor<1x?xf64>
    %v14 = ulm_lam.mul %p.ones, %v13 : tensor<?x1xf64>, tensor<1x?xf64> -> tensor<?x?xf64>
    %v15 = ulm_lam.add %v12, %v14 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
    %v16 = arith.constant -1.0 : f64
    %v17 = ulm_lam.scale %v16, %v11 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
    %v18 = ulm_lam.add %v15, %v17 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
    %v19 = ulm_lam.transpose %v9 : tensor<?x?xf64> -> tensor<?x?xf64>
    %v20 = ulm_lam.mul %v19, %v18 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
    %v21 = ulm_lam.scale %p.step, %v20 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
    %v22 = ulm_lam.scale %v16, %v21 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
    %v23 = ulm_lam.add %v6, %v22 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
    %v24 = ulm_lam.transpose %v18 : tensor<?x?xf64> -> tensor<?x?xf64>
    %v25 = ulm_lam.mul %v24, %p.ones : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    %v26 = ulm_lam.scale %p.step, %v25 : f64, tensor<?x1xf64> -> tensor<?x1xf64>
    %v27 = ulm_lam.scale %v16, %v26 : f64, tensor<?x1xf64> -> tensor<?x1xf64>
    %v28 = ulm_lam.add %v7, %v27 : tensor<?x1xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    scf.yield %v23, %v28 : tensor<?x?xf64>, tensor<?x1xf64>
  }
  %v29 = ulm_lam.transpose %v4#0 : tensor<?x?xf64> -> tensor<?x?xf64>
  return %v29, %v4#1 : tensor<?x?xf64>, tensor<?x1xf64>
}
