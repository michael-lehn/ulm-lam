module {
  func.func @attention(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<?x?xf64> {ulm_lam.dims = [0, 1], ulm_lam.in}, %arg4: tensor<?x?xf64> {ulm_lam.dims = [0, 1], ulm_lam.in}, %arg5: tensor<?x?xf64> {ulm_lam.dims = [0, 2], ulm_lam.in}) -> (tensor<?x?xf64> {ulm_lam.dims = [0, 2], ulm_lam.out}) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.muli %0, %arg0 : index
    %2 = arith.muli %1, %c2 : index
    %3 = arith.muli %arg0, %arg0 : index
    %4 = arith.muli %3, %arg2 : index
    %5 = arith.muli %4, %c2 : index
    %6 = arith.addi %2, %5 : index
    %7 = arith.muli %arg1, %arg0 : index
    %8 = arith.muli %7, %arg2 : index
    %9 = arith.muli %8, %c2 : index
    %10 = arith.muli %arg0, %arg1 : index
    %11 = arith.muli %10, %arg2 : index
    %12 = arith.muli %11, %c2 : index
    %13 = arith.addi %9, %12 : index
    %14 = tensor.empty(%arg0, %arg2) : tensor<?x?xf64>
    %15 = arith.cmpi ule, %6, %13 : index
    %16 = scf.if %15 -> (tensor<?x?xf64>) {
      %17 = tensor.empty(%arg0, %arg0) : tensor<?x?xf64>
      %18 = ulm_lamk.gemm trans_a(none) trans_b(trans) alpha(%cst_0) beta(%cst) ins(%arg3, %arg4 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%17 : tensor<?x?xf64>) -> tensor<?x?xf64>
      %19 = ulm_lamk.gemm trans_a(none) trans_b(none) alpha(%cst_0) beta(%cst) ins(%18, %arg5 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%14 : tensor<?x?xf64>) -> tensor<?x?xf64>
      scf.yield %19 : tensor<?x?xf64>
    } else {
      %17 = tensor.empty(%arg1, %arg2) : tensor<?x?xf64>
      %18 = ulm_lamk.gemm trans_a(trans) trans_b(none) alpha(%cst_0) beta(%cst) ins(%arg4, %arg5 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%17 : tensor<?x?xf64>) -> tensor<?x?xf64>
      %19 = ulm_lamk.gemm trans_a(none) trans_b(none) alpha(%cst_0) beta(%cst) ins(%arg3, %18 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%14 : tensor<?x?xf64>) -> tensor<?x?xf64>
      scf.yield %19 : tensor<?x?xf64>
    }
    return %16 : tensor<?x?xf64>
  }
}

