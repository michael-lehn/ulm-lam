module {
  func.func @gls(%arg0: index, %arg1: index, %arg2: tensor<?x?xf64, #ulm_lam.props<full_rank>> {ulm_lam.dims = [1, 0], ulm_lam.in}, %arg3: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg4: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [1], ulm_lam.out}) {
    %0 = ulm_lam.transpose %arg2 : tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64, #ulm_lam.props<full_rank>>
    %1 = ulm_lam.inverse %arg3 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>
    %2 = ulm_lam.mul %arg2, %1, %0 : tensor<?x?xf64, #ulm_lam.props<full_rank>>, tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>, tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64, #ulm_lam.props<spd>>
    %3 = ulm_lam.inverse %2 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>
    %4 = ulm_lam.inverse %arg3 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>
    %5 = ulm_lam.mul %3, %arg2, %4, %arg4 : tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>, tensor<?x?xf64, #ulm_lam.props<full_rank>>, tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>, tensor<?x1xf64> -> tensor<?x1xf64>
    return %5 : tensor<?x1xf64>
  }
}

