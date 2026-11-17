func.func @gls(%p.X: tensor<2000x200xf64, #ulm_lam.props<full_rank>> {ulm_lam.inout}, %p.M: tensor<2000x2000xf64, #ulm_lam.props<spd>> {ulm_lam.inout}, %p.y: tensor<2000x1xf64> {ulm_lam.inout}) -> (tensor<200x1xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.transpose %p.X : tensor<2000x200xf64, #ulm_lam.props<full_rank>> -> tensor<200x2000xf64>
  %v2 = ulm_lam.inverse %p.M : tensor<2000x2000xf64, #ulm_lam.props<spd>> -> tensor<2000x2000xf64>
  %v3 = ulm_lam.mul %v1, %v2 : tensor<200x2000xf64>, tensor<2000x2000xf64> -> tensor<200x2000xf64>
  %v4 = ulm_lam.mul %v3, %p.X : tensor<200x2000xf64>, tensor<2000x200xf64, #ulm_lam.props<full_rank>> -> tensor<200x200xf64>
  %v5 = ulm_lam.inverse %v4 : tensor<200x200xf64> -> tensor<200x200xf64>
  %v6 = ulm_lam.mul %v5, %v1 : tensor<200x200xf64>, tensor<200x2000xf64> -> tensor<200x2000xf64>
  %v7 = ulm_lam.mul %v6, %v2 : tensor<200x2000xf64>, tensor<2000x2000xf64> -> tensor<200x2000xf64>
  %v8 = ulm_lam.mul %v7, %p.y : tensor<200x2000xf64>, tensor<2000x1xf64> -> tensor<200x1xf64>
  return %v8 : tensor<200x1xf64>
}
