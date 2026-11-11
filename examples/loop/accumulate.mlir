// Matrix operations inside a loop, as the frontend emits them:
//
//   for i = 0, ..., n-1:   B = foo(i);   X += A * B * C
//
// X is passed by reference: the result is marked {ulm_lam.inout = 2}, i.e. it
// is the updated value of argument 2.  B = foo(i) calls an external
// function that returns a matrix.
//
// The pipeline (examples/loop/run-c.sh):
//   ulm-lam-normalize, ulm-lam-infer-properties
//   ulm-lam-select-kernels       A B C by the matrix chain DP; X + A B C fused
//                             into the last gemm (beta = 1, outs = X)
//   ulm-lam-strip-properties
//   ulm-lam-results-to-out-params  foo(i) -> foo(i, B), X by reference
//   one-shot-bufferize, buffer-loop-hoisting, buffer-deallocation-pipeline
//   convert-ulm-lamk-to-emitc    gemm -> cblas_dgemm
//   mlir-translate --mlir-to-cpp, clang-format

!m = tensor<500x500xf64>
func.func private @foo(index) -> !m

func.func @accumulate(%A: !m, %C: !m, %X0: !m, %n: index)
    -> (!m {ulm_lam.inout = 2 : i64}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %X = scf.for %i = %c0 to %n step %c1 iter_args(%X = %X0) -> !m {
    %B  = func.call @foo(%i) : (index) -> !m
    %P  = ulm_lam.mul %A, %B, %C : !m, !m, !m -> !m
    %Xn = ulm_lam.add %X, %P : !m, !m -> !m
    scf.yield %Xn : !m
  }
  return %X : !m
}
