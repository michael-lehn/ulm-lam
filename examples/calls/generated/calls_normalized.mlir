module {
  func.func @bar(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg2: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.out}) {
    %0 = ulm_lam.inverse %arg1 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
    %1 = ulm_lam.mul %0, %arg2 : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
  func.func @foo(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg2: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg3: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.out}) {
    %0 = call @bar(%arg0, %arg2, %arg3) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    %1 = call @bar(%arg0, %arg1, %0) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
}

