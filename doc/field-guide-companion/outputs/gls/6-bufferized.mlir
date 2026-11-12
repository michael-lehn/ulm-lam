module {
  func.func @gls(%arg0: index, %arg1: index, %arg2: memref<?x?xf64> {ulm_lam.dims = [1, 0], ulm_lam.in}, %arg3: memref<?x?xf64> {ulm_lam.dims = [0, 0], ulm_lam.in}, %arg4: memref<?x1xf64> {ulm_lam.dims = [0], ulm_lam.in}, %arg5: memref<?x1xf64> {bufferize.result, ulm_lam.dims = [1]}, %arg6: memref<?xi8> {ulm_lam.workspace}, %arg7: memref<1xi64> {ulm_lam.workspace}) -> i64 attributes {ulm_lam.info = ["Cholesky factorisation of v4: not positive definite", "Cholesky factorisation of a computed matrix: not positive definite"]} {
    %c2_i64 = arith.constant 2 : i64
    %c-8_i64 = arith.constant -8 : i64
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c-64 = arith.constant -64 : index
    %c63 = arith.constant 63 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg7[%c0] : memref<1xi64>
    %1 = arith.muli %arg0, %c8 : index
    %2 = arith.muli %1, %arg0 : index
    %3 = arith.addi %2, %c63 : index
    %4 = arith.andi %3, %c-64 : index
    %5 = arith.muli %arg1, %c8 : index
    %6 = arith.muli %5, %arg1 : index
    %7 = arith.addi %6, %c63 : index
    %8 = arith.andi %7, %c-64 : index
    %9 = arith.addi %4, %8 : index
    %10 = arith.muli %1, %arg1 : index
    %11 = arith.addi %10, %c63 : index
    %12 = arith.andi %11, %c-64 : index
    %13 = arith.addi %9, %12 : index
    %14 = arith.addi %1, %c63 : index
    %15 = arith.andi %14, %c-64 : index
    %16 = arith.addi %13, %15 : index
    %17 = arith.index_cast %16 : index to i64
    %18 = arith.cmpi eq, %0, %c-1_i64 : i64
    %19 = scf.if %18 -> (i64) {
      memref.store %17, %arg7[%c0] : memref<1xi64>
      scf.yield %c0_i64 : i64
    } else {
      %20 = arith.cmpi sgt, %0, %c0_i64 : i64
      %21 = arith.cmpi slt, %0, %17 : i64
      %22 = arith.andi %20, %21 : i1
      %23 = scf.if %22 -> (i64) {
        scf.yield %c-8_i64 : i64
      } else {
        %24 = arith.cmpi eq, %0, %c0_i64 : i64
        %25 = scf.if %24 -> (memref<?xi8>) {
          %alloc = memref.alloc(%16) {alignment = 64 : i64} : memref<?xi8>
          scf.yield %alloc : memref<?xi8>
        } else {
          scf.yield %arg6 : memref<?xi8>
        }
        %view = memref.view %25[%c0][%arg0, %arg0] : memref<?xi8> to memref<?x?xf64>
        memref.copy %arg3, %view : memref<?x?xf64> to memref<?x?xf64>
        %info = ulm_lamk.potrf uplo(lower) outs(%view : memref<?x?xf64>)
        %26 = arith.cmpi ne, %info, %c0_i64 : i64
        %27 = arith.extui %26 : i1 to i64
        %28 = arith.cmpi eq, %27, %c0_i64 : i64
        %29 = scf.if %28 -> (i64) {
          %view_1 = memref.view %25[%4][%arg1, %arg1] : memref<?xi8> to memref<?x?xf64>
          %view_2 = memref.view %25[%9][%arg0, %arg1] : memref<?xi8> to memref<?x?xf64>
          ulm_lamk.gecopy trans(trans) ins(%arg2 : memref<?x?xf64>) outs(%view_2 : memref<?x?xf64>)
          ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit) alpha(%cst) ins(%view : memref<?x?xf64>) outs(%view_2 : memref<?x?xf64>)
          ulm_lamk.syrk uplo(lower) trans(trans) alpha(%cst) beta(%cst_0) ins(%view_2 : memref<?x?xf64>) outs(%view_1 : memref<?x?xf64>)
          %info_3 = ulm_lamk.potrf uplo(lower) outs(%view_1 : memref<?x?xf64>)
          %30 = arith.cmpi ne, %info_3, %c0_i64 : i64
          %31 = arith.select %30, %c2_i64, %c0_i64 : i64
          %32 = arith.cmpi eq, %31, %c0_i64 : i64
          scf.if %32 {
            %view_4 = memref.view %25[%13][%arg0] : memref<?xi8> to memref<?x1xf64>
            memref.copy %arg4, %view_4 : memref<?x1xf64> to memref<?x1xf64>
            ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%view : memref<?x?xf64>) outs(%view_4 : memref<?x1xf64>)
            ulm_lamk.gemv trans(trans) alpha(%cst) beta(%cst_0) ins(%view_2, %view_4 : memref<?x?xf64>, memref<?x1xf64>) outs(%arg5 : memref<?x1xf64>)
            ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%view_1 : memref<?x?xf64>) outs(%arg5 : memref<?x1xf64>)
            ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%view_1 : memref<?x?xf64>) outs(%arg5 : memref<?x1xf64>)
          }
          scf.yield %31 : i64
        } else {
          scf.yield %27 : i64
        }
        scf.if %24 {
          memref.dealloc %25 : memref<?xi8>
        }
        scf.yield %29 : i64
      }
      scf.yield %23 : i64
    }
    return %19 : i64
  }
}

