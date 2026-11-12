module {
  func.func @gls(%arg0: index, %arg1: index, %arg2: tensor<?x?xf64, #ulm_lam.props<full_rank>> {ulm_lam.dims = [1, 0], ulm_lam.in}, %arg3: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg4: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [1], ulm_lam.out}) {
    %result, %info = ulm_lamk.potrf uplo(lower) outs(%arg3 : tensor<?x?xf64, #ulm_lam.props<spd>>) -> tensor<?x?xf64, #ulm_lam.props<spd>>
    %0 = tensor.empty(%arg1, %arg1) : tensor<?x?xf64, #ulm_lam.props<spd>>
    %1 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64>
    %2 = ulm_lamk.gecopy trans(trans) ins(%arg2 : tensor<?x?xf64, #ulm_lam.props<full_rank>>) outs(%1 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %cst = arith.constant 1.000000e+00 : f64
    %3 = ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit) alpha(%cst) ins(%result : tensor<?x?xf64, #ulm_lam.props<spd>>) outs(%2 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %4 = ulm_lamk.syrk uplo(lower) trans(trans) alpha(%cst_0) beta(%cst_1) ins(%3 : tensor<?x?xf64>) outs(%0 : tensor<?x?xf64, #ulm_lam.props<spd>>) -> tensor<?x?xf64, #ulm_lam.props<spd>>
    %5 = ulm_lam.assume %4 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64, #ulm_lam.props<symmetric, spd, lower_stored>>
    %result_2, %info_3 = ulm_lamk.potrf uplo(lower) outs(%5 : tensor<?x?xf64, #ulm_lam.props<symmetric, spd, lower_stored>>) -> tensor<?x?xf64, #ulm_lam.props<symmetric, spd, lower_stored>>
    %6 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result : tensor<?x?xf64, #ulm_lam.props<spd>>) outs(%arg4 : tensor<?x1xf64>) -> tensor<?x1xf64>
    %7 = tensor.empty(%arg1) : tensor<?x1xf64>
    %cst_4 = arith.constant 1.000000e+00 : f64
    %cst_5 = arith.constant 0.000000e+00 : f64
    %8 = ulm_lamk.gemv trans(trans) alpha(%cst_4) beta(%cst_5) ins(%3, %6 : tensor<?x?xf64>, tensor<?x1xf64>) outs(%7 : tensor<?x1xf64>) -> tensor<?x1xf64>
    %9 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result_2 : tensor<?x?xf64, #ulm_lam.props<symmetric, spd, lower_stored>>) outs(%8 : tensor<?x1xf64>) -> tensor<?x1xf64>
    %10 = ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%result_2 : tensor<?x?xf64, #ulm_lam.props<symmetric, spd, lower_stored>>) outs(%9 : tensor<?x1xf64>) -> tensor<?x1xf64>
    return %10 : tensor<?x1xf64>
  }
}

