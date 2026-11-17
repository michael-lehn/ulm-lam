// A result type that claims more than inference can justify: the
// product of two SPD matrices is not symmetric in general.
func.func @f(%A: tensor<3x3xf64, #ulm_lam.props<spd>>,
             %B: tensor<3x3xf64, #ulm_lam.props<spd>>)
    -> tensor<3x3xf64, #ulm_lam.props<spd>> {
  %C = ulm_lam.mul %A, %B : tensor<3x3xf64, #ulm_lam.props<spd>>,
       tensor<3x3xf64, #ulm_lam.props<spd>> -> tensor<3x3xf64, #ulm_lam.props<spd>>
  return %C : tensor<3x3xf64, #ulm_lam.props<spd>>
}
