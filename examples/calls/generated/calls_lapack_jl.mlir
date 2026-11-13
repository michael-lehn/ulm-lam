func.func @bar_(%s.n: index, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.inout, ulm_lam.dims = [0, 0]}, %p.y: tensor<?x1xf64> {ulm_lam.inout, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.inout = 2 : i64, ulm_lam.dims = [0]}) {
  %v1 = ulm_lam.transpose %p.M : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v2 = ulm_lam.inverse %v1 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v3 = ulm_lam.mul %v2, %p.y : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
  return %v3 : tensor<?x1xf64>
}

func.func @foo_(%s.n: index, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.inout, ulm_lam.dims = [0, 0]}, %p.N: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.inout, ulm_lam.dims = [0, 0]}, %p.y: tensor<?x1xf64> {ulm_lam.inout, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.inout = 3 : i64, ulm_lam.dims = [0]}) {
  %v1 = ulm_lam.transpose %p.M : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v2 = ulm_lam.transpose %p.N : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v3 = ulm_lam.transpose %v2 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.assume %v3 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v5 = func.call @bar_(%s.n, %v4, %p.y) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> (tensor<?x1xf64>)
  %v6 = ulm_lam.transpose %v1 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v7 = ulm_lam.assume %v6 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v8 = func.call @bar_(%s.n, %v7, %v5) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> (tensor<?x1xf64>)
  return %v8 : tensor<?x1xf64>
}

func.func @foo_matlab(%s.n: index, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.N: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.y: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [0]}) {
  %v1 = ulm_lam.transpose %p.M : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v2 = ulm_lam.transpose %p.N : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v3 = ulm_lam.transpose %v1 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.assume %v3 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v5 = ulm_lam.transpose %v2 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v6 = ulm_lam.assume %v5 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v7 = func.call @foo_(%s.n, %v4, %v6, %p.y) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> (tensor<?x1xf64>)
  return %v7 : tensor<?x1xf64>
}
