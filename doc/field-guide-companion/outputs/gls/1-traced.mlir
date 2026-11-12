func.func @gls(%s.n: index, %s.m: index, %p.X: tensor<?x?xf64, #ulm_lam.props<full_rank>> {ulm_lam.in, ulm_lam.dims = [1, 0]}, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.y: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [1]}) {
  %v1 = ulm_lam.transpose %p.X : tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64>
  %v2 = ulm_lam.transpose %p.M : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v3 = ulm_lam.transpose %v1 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.inverse %v2 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v5 = ulm_lam.mul %v3, %v4 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v6 = ulm_lam.mul %v5, %v1 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v7 = ulm_lam.inverse %v6 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v8 = ulm_lam.transpose %v1 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v9 = ulm_lam.inverse %v2 : tensor<?x?xf64> -> tensor<?x?xf64>
  %v10 = ulm_lam.mul %v7, %v8 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v11 = ulm_lam.mul %v10, %v9 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v12 = ulm_lam.mul %v11, %p.y : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
  return %v12 : tensor<?x1xf64>
}
