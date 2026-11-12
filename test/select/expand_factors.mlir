// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s
// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels{expand-factors=false}))' | FileCheck %s --check-prefix=OFF

// The covariance update of a Kalman filter, P - K H P with
// K = P H^T S^-1 and S spd: K's product is expanded into the update, which
// is then (H P)^T S^-1 (H P) = W^T W with W = L^-1 H P, computed when K was:
// a syrk instead of a gemm (found by the e-graph experiment).

// CHECK-LABEL: func.func @update
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.syrk
// CHECK-NOT: ulm_lamk.gemm trans_a(none) trans_b(none) alpha(%{{.*}}) beta(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8x3xf64>, tensor<3x8xf64>)
// OFF-LABEL: func.func @update
// OFF-NOT: ulm_lamk.syrk
func.func @update(%p: tensor<8x8xf64, #ulm_lam.props<symmetric>>, %h: tensor<3x8xf64>, %r: tensor<3x3xf64, #ulm_lam.props<spd>>) -> (tensor<8x3xf64>, tensor<8x8xf64>) {
  %ht = ulm_lam.transpose %h : tensor<3x8xf64> -> tensor<8x3xf64>
  %hp = ulm_lam.mul %h, %p, %ht : tensor<3x8xf64>, tensor<8x8xf64, #ulm_lam.props<symmetric>>, tensor<8x3xf64> -> tensor<3x3xf64>
  %s0 = ulm_lam.add %hp, %r : tensor<3x3xf64>, tensor<3x3xf64, #ulm_lam.props<spd>> -> tensor<3x3xf64>
  %s = ulm_lam.assume %s0 : tensor<3x3xf64> -> tensor<3x3xf64, #ulm_lam.props<spd>>
  %si = ulm_lam.inverse %s : tensor<3x3xf64, #ulm_lam.props<spd>> -> tensor<3x3xf64>
  %k = ulm_lam.mul %p, %ht, %si : tensor<8x8xf64, #ulm_lam.props<symmetric>>, tensor<8x3xf64>, tensor<3x3xf64> -> tensor<8x3xf64>
  %khp = ulm_lam.mul %k, %h, %p : tensor<8x3xf64>, tensor<3x8xf64>, tensor<8x8xf64, #ulm_lam.props<symmetric>> -> tensor<8x8xf64>
  %m1 = arith.constant -1.0 : f64
  %neg = ulm_lam.scale %m1, %khp : f64, tensor<8x8xf64> -> tensor<8x8xf64>
  %pn = ulm_lam.add %p, %neg : tensor<8x8xf64, #ulm_lam.props<symmetric>>, tensor<8x8xf64> -> tensor<8x8xf64>
  return %k, %pn : tensor<8x3xf64>, tensor<8x8xf64>
}
