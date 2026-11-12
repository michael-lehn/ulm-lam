func.func @iterate(%s.n: index, %p.M: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.x0: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}, %p.steps: index) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [0]}) {
  %v1 = ulm_lam.transpose %p.M : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v2 = arith.constant 0 : index
  %v3 = arith.constant 1 : index
  %v4 = scf.for %v5 = %v2 to %p.steps step %v3 iter_args(%v6 = %p.x0) -> (tensor<?x1xf64>) {
    %v7 = ulm_lam.inverse %v1 : tensor<?x?xf64> -> tensor<?x?xf64>
    %v8 = ulm_lam.mul %v7, %v6 : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    scf.yield %v8 : tensor<?x1xf64>
  }
  return %v4 : tensor<?x1xf64>
}
