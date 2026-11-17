// A product whose inner dimensions do not match (hand-written verifier).
func.func @f(%A: tensor<3x4xf64>, %B: tensor<5x2xf64>) -> tensor<3x2xf64> {
  %C = ulm_lam.mul %A, %B : tensor<3x4xf64>, tensor<5x2xf64> -> tensor<3x2xf64>
  return %C : tensor<3x2xf64>
}
