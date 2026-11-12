func.func @f(%a: tensor<4x4xf64>) -> tensor<4x4xf64, #ulm_lam.props<spd>> {
  %s = ulm_lam.assume %a : tensor<4x4xf64> -> tensor<4x4xf64, #ulm_lam.props<spd>>
  return %s : tensor<4x4xf64, #ulm_lam.props<spd>>
}
