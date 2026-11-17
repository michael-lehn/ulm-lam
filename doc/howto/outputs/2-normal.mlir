module {
  func.func @gls(%arg0: tensor<2000x200xf64, #ulm_lam.props<full_rank>> {ulm_lam.inout}, %arg1: tensor<2000x2000xf64, #ulm_lam.props<spd>> {ulm_lam.inout}, %arg2: tensor<2000x1xf64> {ulm_lam.inout}) -> (tensor<200x1xf64> {ulm_lam.out}) {
    %0 = ulm_lam.transpose %arg0 : tensor<2000x200xf64, #ulm_lam.props<full_rank>> -> tensor<200x2000xf64>
    %1 = ulm_lam.inverse %arg1 : tensor<2000x2000xf64, #ulm_lam.props<spd>> -> tensor<2000x2000xf64>
    %2 = ulm_lam.mul %0, %1, %arg0 : tensor<200x2000xf64>, tensor<2000x2000xf64>, tensor<2000x200xf64, #ulm_lam.props<full_rank>> -> tensor<200x200xf64>
    %3 = ulm_lam.inverse %2 : tensor<200x200xf64> -> tensor<200x200xf64>
    %4 = ulm_lam.mul %3, %0, %1, %arg2 : tensor<200x200xf64>, tensor<200x2000xf64>, tensor<2000x2000xf64>, tensor<2000x1xf64> -> tensor<200x1xf64>
    return %4 : tensor<200x1xf64>
  }
}

