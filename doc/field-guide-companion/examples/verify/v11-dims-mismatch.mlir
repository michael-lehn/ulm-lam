// Two run-time sizes n (arg 0) and k (arg 1); A is n x k, B is n x k.
// The product A B needs k == n, which the verifier cannot see (both '?').
func.func @f(%n: index, %k: index,
    %a: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]},
    %b: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]})
    -> (tensor<?x?xf64> {ulm_lam.out, ulm_lam.dims = [0, 1]}) {
  %p = ulm_lam.mul %a, %b : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %p : tensor<?x?xf64>
}
