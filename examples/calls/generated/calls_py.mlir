func.func @bar(%s.n: index, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.y: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [0]}) {
  %v1 = ulm_lam.inverse %p.M : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v2 = ulm_lam.mul %v1, %p.y : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
  return %v2 : tensor<?x1xf64>
}

func.func @foo(%s.n: index, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.N: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.y: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [0]}) {
  %v1 = func.call @bar(%s.n, %p.N, %p.y) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> (tensor<?x1xf64>)
  %v2 = func.call @bar(%s.n, %p.M, %v1) : (index, tensor<?x?xf64, #ulm_lam.props<spd>>, tensor<?x1xf64>) -> (tensor<?x1xf64>)
  return %v2 : tensor<?x1xf64>
}
