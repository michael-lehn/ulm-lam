module {
  func.func @gls(%arg0: memref<2000x200xf64> {ulm_lam.inout}, %arg1: memref<2000x2000xf64> {ulm_lam.inout}, %arg2: memref<2000x1xf64> {ulm_lam.inout}, %arg3: memref<200x1xf64> {bufferize.result}, %arg4: memref<?xi8> {ulm_lam.workspace}, %arg5: memref<1xi64> {ulm_lam.workspace}) -> i64 attributes {ulm_lam.info = ["Cholesky factorisation of v2: not positive definite", "Cholesky factorisation of a computed matrix: not positive definite"]} {
    %c2_i64 = arith.constant 2 : i64
    %c-6_i64 = arith.constant -6 : i64
    %c320000_i64 = arith.constant 320000 : i64
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg5[%c0] : memref<1xi64>
    %1 = arith.cmpi eq, %0, %c-1_i64 : i64
    %2 = scf.if %1 -> (i64) {
      memref.store %c320000_i64, %arg5[%c0] : memref<1xi64>
      scf.yield %c0_i64 : i64
    } else {
      %3 = arith.cmpi sgt, %0, %c0_i64 : i64
      %4 = arith.cmpi slt, %0, %c320000_i64 : i64
      %5 = arith.andi %3, %4 : i1
      %6 = scf.if %5 -> (i64) {
        scf.yield %c-6_i64 : i64
      } else {
        %7 = arith.cmpi eq, %0, %c0_i64 : i64
        %8 = scf.if %7 -> (memref<?xi8>) {
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<320000xi8>
          %cast = memref.cast %alloc : memref<320000xi8> to memref<?xi8>
          scf.yield %cast : memref<?xi8>
        } else {
          scf.yield %arg4 : memref<?xi8>
        }
        %info = ulm_lamk.potrf uplo(lower) outs(%arg1 : memref<2000x2000xf64>)
        %9 = arith.cmpi ne, %info, %c0_i64 : i64
        %10 = arith.extui %9 : i1 to i64
        %11 = arith.cmpi eq, %10, %c0_i64 : i64
        %12 = scf.if %11 -> (i64) {
          %view = memref.view %8[%c0][] : memref<?xi8> to memref<200x200xf64>
          ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit) alpha(%cst) ins(%arg1 : memref<2000x2000xf64>) outs(%arg0 : memref<2000x200xf64>)
          ulm_lamk.syrk uplo(lower) trans(trans) alpha(%cst) beta(%cst_0) ins(%arg0 : memref<2000x200xf64>) outs(%view : memref<200x200xf64>)
          %info_1 = ulm_lamk.potrf uplo(lower) outs(%view : memref<200x200xf64>)
          %13 = arith.cmpi ne, %info_1, %c0_i64 : i64
          %14 = arith.select %13, %c2_i64, %c0_i64 : i64
          %15 = arith.cmpi eq, %14, %c0_i64 : i64
          scf.if %15 {
            ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%arg1 : memref<2000x2000xf64>) outs(%arg2 : memref<2000x1xf64>)
            ulm_lamk.gemv trans(trans) alpha(%cst) beta(%cst_0) ins(%arg0, %arg2 : memref<2000x200xf64>, memref<2000x1xf64>) outs(%arg3 : memref<200x1xf64>)
            ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%view : memref<200x200xf64>) outs(%arg3 : memref<200x1xf64>)
            ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%view : memref<200x200xf64>) outs(%arg3 : memref<200x1xf64>)
          }
          scf.yield %14 : i64
        } else {
          scf.yield %10 : i64
        }
        scf.if %7 {
          memref.dealloc %8 : memref<?xi8>
        }
        scf.yield %12 : i64
      }
      scf.yield %6 : i64
    }
    return %2 : i64
  }
  func.func @gls_keep(%arg0: memref<2000x200xf64> {ulm_lam.in}, %arg1: memref<2000x2000xf64> {ulm_lam.in}, %arg2: memref<2000x1xf64> {ulm_lam.in}, %arg3: memref<200x1xf64> {bufferize.result}, %arg4: memref<?xi8> {ulm_lam.workspace}, %arg5: memref<1xi64> {ulm_lam.workspace}) -> i64 attributes {ulm_lam.info = ["Cholesky factorisation of v2: not positive definite", "Cholesky factorisation of a computed matrix: not positive definite"], ulm_lam.wraps = "gls"} {
    %c-6_i64 = arith.constant -6 : i64
    %c35216000 = arith.constant 35216000 : index
    %c35216000_i64 = arith.constant 35216000 : i64
    %c35200000 = arith.constant 35200000 : index
    %c3200000 = arith.constant 3200000 : index
    %c0_i64 = arith.constant 0 : i64
    %c-64_i64 = arith.constant -64 : i64
    %c63_i64 = arith.constant 63 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg5[%c0] : memref<1xi64>
    %view = memref.view %arg4[%c0][] : memref<?xi8> to memref<2000x200xf64>
    %view_0 = memref.view %arg4[%c1][] : memref<?xi8> to memref<2000x2000xf64>
    %view_1 = memref.view %arg4[%c2][] : memref<?xi8> to memref<2000x1xf64>
    %view_2 = memref.view %arg4[%c3][] : memref<?xi8> to memref<200x1xf64>
    memref.store %c-1_i64, %arg5[%c0] : memref<1xi64>
    %1 = call @gls(%view, %view_0, %view_1, %view_2, %arg4, %arg5) {ulm_lam.query} : (memref<2000x200xf64>, memref<2000x2000xf64>, memref<2000x1xf64>, memref<200x1xf64>, memref<?xi8>, memref<1xi64>) -> i64
    %2 = arith.cmpi eq, %1, %c0_i64 : i64
    %3 = scf.if %2 -> (i64) {
      %4 = memref.load %arg5[%c0] : memref<1xi64>
      %5 = arith.addi %4, %c63_i64 : i64
      %6 = arith.andi %5, %c-64_i64 : i64
      %7 = arith.addi %6, %c35216000_i64 : i64
      memref.store %0, %arg5[%c0] : memref<1xi64>
      %8 = arith.cmpi eq, %0, %c-1_i64 : i64
      %9 = scf.if %8 -> (i64) {
        memref.store %7, %arg5[%c0] : memref<1xi64>
        scf.yield %c0_i64 : i64
      } else {
        %10 = arith.cmpi sgt, %0, %c0_i64 : i64
        %11 = arith.cmpi slt, %0, %7 : i64
        %12 = arith.andi %10, %11 : i1
        %13 = scf.if %12 -> (i64) {
          scf.yield %c-6_i64 : i64
        } else {
          %14 = arith.cmpi eq, %0, %c0_i64 : i64
          %15 = scf.if %14 -> (memref<?xi8>) {
            %19 = arith.index_cast %7 : i64 to index
            %alloc = memref.alloc(%19) {alignment = 64 : i64} : memref<?xi8>
            scf.yield %alloc : memref<?xi8>
          } else {
            scf.yield %arg4 : memref<?xi8>
          }
          %view_3 = memref.view %15[%c0][] : memref<?xi8> to memref<2000x200xf64>
          memref.copy %arg0, %view_3 : memref<2000x200xf64> to memref<2000x200xf64>
          %view_4 = memref.view %15[%c3200000][] : memref<?xi8> to memref<2000x2000xf64>
          memref.copy %arg1, %view_4 : memref<2000x2000xf64> to memref<2000x2000xf64>
          %view_5 = memref.view %15[%c35200000][] : memref<?xi8> to memref<2000x1xf64>
          memref.copy %arg2, %view_5 : memref<2000x1xf64> to memref<2000x1xf64>
          %16 = arith.index_cast %4 : i64 to index
          %view_6 = memref.view %15[%c35216000][%16] : memref<?xi8> to memref<?xi8>
          memref.store %4, %arg5[%c0] : memref<1xi64>
          %17 = func.call @gls(%view_3, %view_4, %view_5, %arg3, %view_6, %arg5) : (memref<2000x200xf64>, memref<2000x2000xf64>, memref<2000x1xf64>, memref<200x1xf64>, memref<?xi8>, memref<1xi64>) -> i64
          %18 = arith.cmpi eq, %17, %c0_i64 : i64
          scf.if %18 {
            memref.store %0, %arg5[%c0] : memref<1xi64>
          }
          scf.if %14 {
            memref.dealloc %15 : memref<?xi8>
          }
          scf.yield %17 : i64
        }
        scf.yield %13 : i64
      }
      scf.yield %9 : i64
    } else {
      scf.yield %1 : i64
    }
    return %3 : i64
  }
}

