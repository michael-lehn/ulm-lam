module {
  func.func @gls(%arg0: tensor<2000x200xf64, #ulm_lam.props<full_rank>> {ulm_lam.inout}, %arg1: tensor<2000x2000xf64, #ulm_lam.props<spd>> {ulm_lam.inout}, %arg2: tensor<2000x1xf64> {ulm_lam.inout}) -> (tensor<200x1xf64> {ulm_lam.out}) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %result, %info = ulm_lamk.potrf uplo(lower) outs(%arg1 : tensor<2000x2000xf64, #ulm_lam.props<spd>>) -> tensor<2000x2000xf64, #ulm_lam.props<spd>>
    %0 = tensor.empty() : tensor<200x200xf64, #ulm_lam.props<spd>>
    %1 = ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit) alpha(%cst_0) ins(%result : tensor<2000x2000xf64, #ulm_lam.props<spd>>) outs(%arg0 : tensor<2000x200xf64, #ulm_lam.props<full_rank>>) -> tensor<2000x200xf64, #ulm_lam.props<full_rank>>
    %2 = ulm_lamk.syrk uplo(lower) trans(trans) alpha(%cst_0) beta(%cst) ins(%1 : tensor<2000x200xf64, #ulm_lam.props<full_rank>>) outs(%0 : tensor<200x200xf64, #ulm_lam.props<spd>>) -> tensor<200x200xf64, #ulm_lam.props<spd>>
    %3 = ulm_lam.assume %2 : tensor<200x200xf64, #ulm_lam.props<spd>> -> tensor<200x200xf64, #ulm_lam.props<symmetric, spd, lower_stored>>
    %result_1, %info_2 = ulm_lamk.potrf uplo(lower) outs(%3 : tensor<200x200xf64, #ulm_lam.props<symmetric, spd, lower_stored>>) -> tensor<200x200xf64, #ulm_lam.props<symmetric, spd, lower_stored>>
    %4 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result : tensor<2000x2000xf64, #ulm_lam.props<spd>>) outs(%arg2 : tensor<2000x1xf64>) -> tensor<2000x1xf64>
    %5 = tensor.empty() : tensor<200x1xf64>
    %6 = ulm_lamk.gemv trans(trans) alpha(%cst_0) beta(%cst) ins(%1, %4 : tensor<2000x200xf64, #ulm_lam.props<full_rank>>, tensor<2000x1xf64>) outs(%5 : tensor<200x1xf64>) -> tensor<200x1xf64>
    %7 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result_1 : tensor<200x200xf64, #ulm_lam.props<symmetric, spd, lower_stored>>) outs(%6 : tensor<200x1xf64>) -> tensor<200x1xf64>
    %8 = ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%result_1 : tensor<200x200xf64, #ulm_lam.props<symmetric, spd, lower_stored>>) outs(%7 : tensor<200x1xf64>) -> tensor<200x1xf64>
    return %8 : tensor<200x1xf64>
  }
}

