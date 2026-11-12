func.func @f(%a: tensor<3x4xf64>) -> tensor<3x4xf64> {
  %i = ulm_lam.inverse %a : tensor<3x4xf64> -> tensor<3x4xf64>
  return %i : tensor<3x4xf64>
}
