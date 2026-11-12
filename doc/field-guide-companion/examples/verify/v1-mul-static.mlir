func.func @f(%a: tensor<3x4xf64>, %b: tensor<5x2xf64>) -> tensor<3x2xf64> {
  %p = ulm_lam.mul %a, %b : tensor<3x4xf64>, tensor<5x2xf64> -> tensor<3x2xf64>
  return %p : tensor<3x2xf64>
}
