// An integer matrix (generated verifier: the operand constraint).
func.func @f(%A: tensor<3x3xi32>) -> tensor<3x3xi32> {
  %T = ulm_lam.transpose %A : tensor<3x3xi32> -> tensor<3x3xi32>
  return %T : tensor<3x3xi32>
}
