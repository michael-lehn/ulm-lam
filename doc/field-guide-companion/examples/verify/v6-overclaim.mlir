func.func @f(%a: tensor<4x3xf64>) -> tensor<3x3xf64, #ulm_lam.props<spd>> {
  %at = ulm_lam.transpose %a : tensor<4x3xf64> -> tensor<3x4xf64>
  %g = ulm_lam.mul %at, %a : tensor<3x4xf64>, tensor<4x3xf64> -> tensor<3x3xf64, #ulm_lam.props<spd>>
  return %g : tensor<3x3xf64, #ulm_lam.props<spd>>
}
