//===- UlmLamPasses.td - ulm_lam pipeline passes -----------*- tablegen -*-===//

#ifndef ULM_LAM_PASSES_TD
#define ULM_LAM_PASSES_TD

include "mlir/Pass/PassBase.td"

def UlmLamNormalize : Pass<"ulm-lam-normalize", "func::FuncOp"> {
  let summary = "Bring ulm_lam expressions into normal form";
  let description = [{
    Greedy application of the ulm_lam canonicalisation patterns: flatten n-ary
    mul/add, push transposes and inverses to the leaves, hoist scales,
    remove identities, fold double transposes and transposes of symmetric
    matrices.  Followed by CSE.  Terminates because every pattern strictly
    decreases a well-founded measure (depth of transpose/inverse nodes,
    nesting depth of mul/add).
  }];
  let dependentDialects = ["ulm_lam::UlmLamDialect"];
}

def UlmLamInferProperties : Pass<"ulm-lam-infer-properties", "func::FuncOp"> {
  let summary = "Refine #ulm_lam.props encodings to the strongest inferable";
  let description = [{
    Walks the function in topological order and replaces every result type
    by the type returned by InferTypeOpInterface.  Function signatures are
    kept; a `ulm_lam.assume` bridges refined values to unrefined returns.
  }];
}

def UlmLamSelectKernels : Pass<"ulm-lam-select-kernels", "func::FuncOp"> {
  let summary = "Map ulm_lam expressions onto ulm_lamk kernels (solve the LAMP)";
  let description = [{
    Strategies:
      gmc    -- implemented (lib/UlmLamPasses.cpp).  Every inverted matrix is
                factorised once by its properties (triangular: itself; spd:
                potrf; otherwise: getrf), right after its definition.  Each
                product is parenthesised by the generalised matrix chain
                dynamic programme over gemm/gemv, left solves (trsm, trsv,
                getrs), right triangular solves and Z^T Z.  Sums accumulate
                into their first plain term (gescal, geaxpy, gemm with
                beta = 1).
      search -- Linnea-style best-first search over derivations, using
                gmc as inner solver; anytime, bounded by `budget-ms`
      egraph -- equality saturation (eqsat dialect / egglog) with
                property lattices and FLOP-cost extraction (experimental)
  }];
  let options = [
    Option<"strategy", "strategy", "std::string", /*default=*/"\"gmc\"",
           "gmc | search | egraph">,
    Option<"budgetMs", "budget-ms", "unsigned", /*default=*/"1000",
           "Time budget for the search strategy">,
    Option<"costModel", "cost-model", "std::string", /*default=*/"\"flops\"",
           "flops | measured">
  ];
  let dependentDialects = ["ulm_lamk::UlmLamkDialect", "tensor::TensorDialect",
                           "arith::ArithDialect"];
}

def UlmLamResultsToOutParams : Pass<"ulm-lam-results-to-out-params", "ModuleOp"> {
  let summary = "Pass matrix results by reference (destination-passing calls)";
  let description = [{
    Runs before bufferization.  Implemented in lib/UlmLamResultsToOutParams.cpp.
      * External declarations returning static tensors get one memref out
        parameter per result; every call `X = foo(i)` becomes
        `memref.alloc` + `foo(i, X)` + `bufferization.to_tensor restrict
        writable`.  buffer-loop-hoisting later moves the allocation out of
        loops.
      * A result marked `{ulm_lam.inout = i}` is the updated value of argument
        i: the argument becomes a memref (read via to_tensor), the result is
        written back with `bufferization.materialize_in_destination` and
        dropped, so the matrix is updated in place by reference.
      * Other tensor arguments of public functions (passed by value) get
        `bufferization.writable = false`: the caller's matrix is never
        overwritten; bufferization copies where a kernel works in place.
  }];
  let dependentDialects = ["memref::MemRefDialect",
                           "bufferization::BufferizationDialect"];
}

def UlmLamWorkspace : Pass<"ulm-lam-workspace", "ModuleOp"> {
  let summary = "Temporaries from a caller-provided workspace (LAPACK style)";
  let description = [{
    Runs after bufferization.  Implemented in lib/UlmLamWorkspace.cpp.  A
    public function with temporaries gets two trailing parameters,
    `int8_t *work, int64_t *workspace_info` (sizes in bytes):
      -1  query: set *workspace_info to the required size and return
       0  allocate one block of the required size, free it at the end
      >0  the size of work; too small: call
          ulm_lam_workspace_error(required, given) and return
    All temporaries become views of the one block at 64-byte aligned
    offsets computed at compile time.  Functions without temporaries keep
    their signature.
  }];
  let dependentDialects = ["memref::MemRefDialect", "scf::SCFDialect",
                           "arith::ArithDialect", "func::FuncDialect"];
}

def UlmLamStripProperties : Pass<"ulm-lam-strip-properties", "ModuleOp"> {
  let summary = "Drop #ulm_lam.props encodings before bufferization";
  let description = [{
    After kernel selection the properties are encoded in kernel flags
    (uplo, diag, ...).  Encodings are removed so that standard tensor
    passes and One-Shot Bufferize see plain tensors.
  }];
}

def ConvertUlmLamkToUlmBLAS : Pass<"convert-ulm-lamk-to-ulmblas", "ModuleOp"> {
  let summary = "Lower bufferized ulm_lamk kernels to C++ calling ulmBLAS";
  let description = [{
    Implemented in lib/ConvertUlmLamkToUlmBLAS.cpp.  Every kernel becomes an
    `emitc.call_opaque` of the header-only ulmBLAS C++ interface
    (ulmBLAS::gemm, gemv, trlsm/trusm, trlsv/trusv, geaxpy, gescal, potrf,
    getrf, getrs).  Static row-major memrefs become pointers with strides
    (incRow, incCol) = (columns, 1); a transposed operand is the same
    pointer with swapped strides, so trans and side attributes disappear.
    Functions get extern "C".  Run with the upstream
    convert-{func,arith,scf}-to-emitc patterns (included); then
    `mlir-translate --mlir-to-cpp` prints C++17.
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
}

def ConvertUlmLamkToLinalg : Pass<"convert-ulm-lamk-to-linalg", "func::FuncOp"> {
  let summary = "Reference lowering of ulm_lamk to linalg/scf";
  let description = [{
    Library-free lowering (gemm -> linalg.matmul, trsm/potrf -> scf loops).
    Used for differential testing against the runtime lowering and as a
    fallback where no BLAS is available.
  }];
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect"];
}

#endif // ULM_LAM_PASSES_TD
