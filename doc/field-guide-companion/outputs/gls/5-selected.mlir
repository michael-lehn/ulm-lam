module {
  func.func @gls(%arg0: index, %arg1: index, %arg2: tensor<?x?xf64> {ulm_lam.dims = [1, 0], ulm_lam.in}, %arg3: tensor<?x?xf64> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg4: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [1], ulm_lam.out}) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %result, %info = ulm_lamk.potrf uplo(lower) outs(%arg3 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %0 = tensor.empty(%arg1, %arg1) : tensor<?x?xf64>
    %1 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64>
    %2 = ulm_lamk.gecopy trans(trans) ins(%arg2 : tensor<?x?xf64>) outs(%1 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %3 = ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit) alpha(%cst_0) ins(%result : tensor<?x?xf64>) outs(%2 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %4 = ulm_lamk.syrk uplo(lower) trans(trans) alpha(%cst_0) beta(%cst) ins(%3 : tensor<?x?xf64>) outs(%0 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %result_1, %info_2 = ulm_lamk.potrf uplo(lower) outs(%4 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %5 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result : tensor<?x?xf64>) outs(%arg4 : tensor<?x1xf64>) -> tensor<?x1xf64>
    %6 = tensor.empty(%arg1) : tensor<?x1xf64>
    %7 = ulm_lamk.gemv trans(trans) alpha(%cst_0) beta(%cst) ins(%3, %5 : tensor<?x?xf64>, tensor<?x1xf64>) outs(%6 : tensor<?x1xf64>) -> tensor<?x1xf64>
    %8 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result_1 : tensor<?x?xf64>) outs(%7 : tensor<?x1xf64>) -> tensor<?x1xf64>
    %9 = ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%result_1 : tensor<?x?xf64>) outs(%8 : tensor<?x1xf64>) -> tensor<?x1xf64>
    return %9 : tensor<?x1xf64>
  }
}

