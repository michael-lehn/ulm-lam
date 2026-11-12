// A Cholesky factorisation whose results (factor and info) are unused.
func.func @f(%a: tensor<4x4xf64>, %b: memref<4x4xf64>) -> tensor<4x4xf64> {
  %l, %info = ulm_lamk.potrf uplo(lower) outs(%a : tensor<4x4xf64>) -> tensor<4x4xf64>
  %info2 = ulm_lamk.potrf uplo(lower) outs(%b : memref<4x4xf64>)
  return %a : tensor<4x4xf64>
}
