// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels{lapack=true}))' | FileCheck %s --check-prefix=LAPACK
// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s --check-prefix=NOLAPACK

// If the backend has LAPACK, L D L^T (getrf/getrs {symmetric}) for a
// symmetric matrix not known to be spd and least squares by QR (gels);
// otherwise LU, and the normal equations with a Cholesky factorisation.

// LAPACK-LABEL: func.func @sym
// LAPACK: ulm_lamk.getrf {{.*}}{symmetric}
// LAPACK: ulm_lamk.getrs {{.*}}{symmetric}
// NOLAPACK-LABEL: func.func @sym
// NOLAPACK-NOT: symmetric}
// NOLAPACK: ulm_lamk.getrf
func.func @sym(%s: tensor<6x6xf64, #ulm_lam.props<symmetric, non_singular>>, %b: tensor<6x2xf64>) -> tensor<6x2xf64> {
  %si = ulm_lam.inverse %s : tensor<6x6xf64, #ulm_lam.props<symmetric, non_singular>> -> tensor<6x6xf64>
  %x = ulm_lam.mul %si, %b : tensor<6x6xf64>, tensor<6x2xf64> -> tensor<6x2xf64>
  return %x : tensor<6x2xf64>
}

// LAPACK-LABEL: func.func @ls
// LAPACK: ulm_lamk.gels
// LAPACK: ulm_lamk.gecopy
// NOLAPACK-LABEL: func.func @ls
// NOLAPACK-NOT: gels
// NOLAPACK: ulm_lamk.syrk
// NOLAPACK: ulm_lamk.potrf
func.func @ls(%a: tensor<20x4xf64, #ulm_lam.props<full_rank>>, %y: tensor<20x1xf64>) -> tensor<4x1xf64> {
  %x = ulm_lam.lstsq %a, %y : tensor<20x4xf64, #ulm_lam.props<full_rank>>, tensor<20x1xf64> -> tensor<4x1xf64>
  return %x : tensor<4x1xf64>
}
