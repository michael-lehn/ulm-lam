module {
  func.func @iterate(%arg0: index, %arg1: tensor<?x?xf64> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg2: tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}, %arg3: index) -> (tensor<?x1xf64> {ulm_lam.dims = [0], ulm_lam.out}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result, %info = ulm_lamk.potrf uplo(lower) outs(%arg1 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %0 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %arg2) -> (tensor<?x1xf64>) {
      %1 = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%result : tensor<?x?xf64>) outs(%arg5 : tensor<?x1xf64>) -> tensor<?x1xf64>
      %2 = ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%result : tensor<?x?xf64>) outs(%1 : tensor<?x1xf64>) -> tensor<?x1xf64>
      %3 = bufferization.materialize_in_destination %2 in %arg5 : (tensor<?x1xf64>, tensor<?x1xf64>) -> tensor<?x1xf64>
      scf.yield %3 : tensor<?x1xf64>
    }
    return %0 : tensor<?x1xf64>
  }
}

