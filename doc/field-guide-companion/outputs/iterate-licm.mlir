module {
  func.func @iterate(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg2: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}, %arg3: index) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.out}) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ulm_lam.inverse %arg1 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>
    %1 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %arg2) -> (tensor<?x1xf64>) {
      %2 = ulm_lam.mul %0, %arg5 : tensor<?x?xf64, #ulm_lam.props<square, symmetric, spd, non_singular, full_rank>>, tensor<?x1xf64> -> tensor<?x1xf64>
      scf.yield %2 : tensor<?x1xf64>
    }
    return %1 : tensor<?x1xf64>
  }
}

