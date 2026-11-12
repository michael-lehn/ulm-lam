func.func @f(%a: tensor<3x4xf64>, %b: tensor<3x2xf64>, %c: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %one = arith.constant 1.0 : f64
  %r = ulm_lamk.gemm trans_a(none) trans_b(none) alpha(%one) beta(%one) ins(%a, %b : tensor<3x4xf64>, tensor<3x2xf64>) outs(%c : tensor<3x2xf64>) -> tensor<3x2xf64>
  return %r : tensor<3x2xf64>
}
