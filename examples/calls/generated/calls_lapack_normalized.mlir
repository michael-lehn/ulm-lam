module {
  func.func @bar_(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.inout}, %arg2: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout = 2 : i64}) {
    %0 = ulm_lam.inverse %arg1 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
    %1 = ulm_lam.mul %0, %arg2 : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
  func.func @foo_(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.inout}, %arg2: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.inout}, %arg3: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.inout = 3 : i64}) {
    %0 = call @bar_(%arg0, %arg2, %arg3) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    %1 = call @bar_(%arg0, %arg1, %0) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
  func.func @foo_matlab(%arg0: index, %arg1: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg2: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg3: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.out}) {
    %0 = call @foo_(%arg0, %arg1, %arg2, %arg3) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> tensor<?x1xf64>
    return %0 : tensor<?x1xf64>
  }
}

