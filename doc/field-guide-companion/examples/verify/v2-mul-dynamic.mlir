func.func @f(%a: tensor<?x?xf64>, %b: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %p = ulm_lam.mul %a, %b : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %p : tensor<?x?xf64>
}
