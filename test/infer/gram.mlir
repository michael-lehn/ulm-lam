// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties))' | FileCheck %s

// X^T M^-1 X is spd when M is spd and X has full rank.

// CHECK-LABEL: func.func @gram
// CHECK: ulm_lam.mul {{.*}} -> tensor<?x?xf64, #ulm_lam.props<{{.*}}spd{{.*}}>>
func.func @gram(%x: tensor<?x?xf64, #ulm_lam.props<full_rank>>, %m: tensor<?x?xf64, #ulm_lam.props<spd>>) -> tensor<?x?xf64> {
  %xt = ulm_lam.transpose %x : tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64>
  %mi = ulm_lam.inverse %m : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %p = ulm_lam.mul %xt, %mi, %x : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64>
  return %p : tensor<?x?xf64>
}
