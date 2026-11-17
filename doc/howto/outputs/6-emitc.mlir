module {
  emitc.include <"cstddef">
  emitc.include <"cstdint">
  emitc.include <"cstdio">
  emitc.include <"cstdlib">
  emitc.include <"cstring">
  emitc.include <"ulmblas/ulmblas.h">
  emitc.include <"ulmblas/lapack/potrf.h">
  emitc.include <"ulmblas/lapack/trtrs.h">
  emitc.verbatim "static void\0Aulm_lam_copy(void *dst, const void *src, std::size_t bytes)\0A{\0A#ifdef _OPENMP\0A    const std::size_t chunk = std::size_t(1) << 20;\0A    if (bytes >= 8 * chunk) {\0A        std::ptrdiff_t chunks = (bytes + chunk - 1) / chunk;\0A#pragma omp parallel for schedule(static)\0A        for (std::ptrdiff_t c = 0; c < chunks; ++c) {\0A            std::size_t off = c * chunk;\0A            std::size_t len = bytes - off < chunk ? bytes - off : chunk;\0A            std::memcpy(static_cast<char *>(dst) + off,\0A                        static_cast<const char *>(src) + off, len);\0A        }\0A        return;\0A    }\0A#endif\0A    std::memcpy(dst, src, bytes);\0A}\0A"
  emitc.verbatim "/* gls: matrices by reference (row-major), owned by the caller\0A *   v1  double[2000][200]   inout  may be overwritten (unspecified on exit)\0A *   v2  double[2000][2000]  inout  may be overwritten (unspecified on exit)\0A *   v3  double[2000][1]     inout  may be overwritten (unspecified on exit)\0A *   v4  double[200][1]      out    written\0A *   v5  int8_t *            work   workspace (bytes)\0A *   v6  int64_t *           info   -1: query, 0: allocate, >0: size of work\0A * returns info:\0A *     0  success\0A *     1  Cholesky factorisation of v2: not positive definite\0A *     2  Cholesky factorisation of a computed matrix: not positive definite\0A *    -6  v6: workspace too small\0A * (info != 0: the outputs are unspecified)\0A */"
  emitc.func @gls(%arg0: !emitc.ptr<f64> {ulm_lam.inout}, %arg1: !emitc.ptr<f64> {ulm_lam.inout}, %arg2: !emitc.ptr<f64> {ulm_lam.inout}, %arg3: !emitc.ptr<f64> {bufferize.result}, %arg4: !emitc.ptr<i8> {ulm_lam.workspace}, %arg5: !emitc.ptr<i64> {ulm_lam.workspace}) -> i64 attributes {specifiers = ["extern \22C\22"], ulm_lam.info = ["Cholesky factorisation of v2: not positive definite", "Cholesky factorisation of a computed matrix: not positive definite"]} {
    %0 = expression  : () -> i64 {
      %10 = "emitc.constant"() <{value = -6 : i64}> : () -> i64
      yield %10 : i64
    }
    %1 = expression  : () -> i64 {
      %10 = "emitc.constant"() <{value = 320000 : i64}> : () -> i64
      yield %10 : i64
    }
    %2 = expression  : () -> f64 {
      %10 = "emitc.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
      yield %10 : f64
    }
    %3 = expression  : () -> f64 {
      %10 = "emitc.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      yield %10 : f64
    }
    %4 = expression  : () -> i64 {
      %10 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
      yield %10 : i64
    }
    %5 = expression  : () -> !emitc.size_t {
      %10 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
      yield %10 : !emitc.size_t
    }
    %6 = expression %arg5, %5 : (!emitc.ptr<i64>, !emitc.size_t) -> i64 {
      %10 = subscript %arg5[%5] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
      %11 = load %10 : <i64>
      yield %11 : i64
    }
    %7 = expression %6 : (i64) -> i1 {
      %10 = "emitc.constant"() <{value = -1 : i64}> : () -> i64
      %11 = cmp eq, %6, %10 : (i64, i64) -> i1
      yield %11 : i1
    }
    %8 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i64>
    if %7 {
      %10 = expression %arg5, %5 : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64> {
        %11 = subscript %arg5[%5] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
        yield %11 : !emitc.lvalue<i64>
      }
      assign %1 : i64 to %10 : <i64>
      assign %4 : i64 to %8 : <i64>
    } else {
      %10 = expression %6, %4, %1 : (i64, i64, i64) -> i1 {
        %13 = cmp lt, %6, %1 : (i64, i64) -> i1
        %14 = cmp gt, %6, %4 : (i64, i64) -> i1
        %15 = bitwise_and %14, %13 : (i1, i1) -> i1
        yield %15 : i1
      }
      %11 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i64>
      if %10 {
        assign %0 : i64 to %11 : <i64>
      } else {
        %13 = expression %6, %4 : (i64, i64) -> i1 {
          %23 = cmp eq, %6, %4 : (i64, i64) -> i1
          yield %23 : i1
        }
        %14 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<i8>>
        if %13 {
          %23 = expression  : () -> !emitc.ptr<!emitc.opaque<"void">> {
            %25 = call_opaque "aligned_alloc"() <{args = [64, 320000]}> : () -> !emitc.ptr<!emitc.opaque<"void">>
            yield %25 : !emitc.ptr<!emitc.opaque<"void">>
          }
          %24 = expression %23 : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<i8> {
            %25 = cast %23 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i8>
            yield %25 : !emitc.ptr<i8>
          }
          assign %24 : !emitc.ptr<i8> to %14 : <!emitc.ptr<i8>>
        } else {
          assign %arg4 : !emitc.ptr<i8> to %14 : <!emitc.ptr<i8>>
        }
        %15 = expression %14 : (!emitc.lvalue<!emitc.ptr<i8>>) -> !emitc.ptr<i8> {
          %23 = load %14 : <!emitc.ptr<i8>>
          yield %23 : !emitc.ptr<i8>
        }
        %16 = expression %arg1 : (!emitc.ptr<f64>) -> i64 {
          %23 = call_opaque "ulmBLAS::potrf<std::ptrdiff_t>"(%arg1) <{args = [#emitc.opaque<"true">, 2000 : i32, 0 : index, 2000 : i32, 1 : i32]}> : (!emitc.ptr<f64>) -> i64
          yield %23 : i64
        }
        %17 = expression %16, %4 : (i64, i64) -> ui1 {
          %23 = cmp ne, %16, %4 : (i64, i64) -> i1
          %24 = cast %23 : i1 to ui1
          yield %24 : ui1
        }
        %18 = expression %17 : (ui1) -> ui64 {
          %23 = cast %17 : ui1 to ui64
          yield %23 : ui64
        }
        %19 = expression %18 : (ui64) -> i64 {
          %23 = cast %18 : ui64 to i64
          yield %23 : i64
        }
        %20 = expression %19, %4 : (i64, i64) -> i1 {
          %23 = cmp eq, %19, %4 : (i64, i64) -> i1
          yield %23 : i1
        }
        %21 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i64>
        if %20 {
          %23 = expression %5 : (!emitc.size_t) -> i64 {
            %28 = cast %5 : !emitc.size_t to i64
            yield %28 : i64
          }
          %24 = expression %15, %23 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
            %28 = add %15, %23 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
            %29 = cast %28 : !emitc.ptr<i8> to !emitc.ptr<f64>
            yield %29 : !emitc.ptr<f64>
          }
          call_opaque "(void)ulmBLAS::trtrs<std::ptrdiff_t>"(%arg1, %arg0) <{args = [#emitc.opaque<"true">, #emitc.opaque<"false">, #emitc.opaque<"false">, #emitc.opaque<"false">, 2000 : i32, 200 : i32, 0 : index, 2000 : i32, 1 : i32, 1 : index, 200 : i32, 1 : i32]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
          call_opaque "ulmBLAS::sylrk<std::ptrdiff_t>"(%2, %arg0, %3, %24) <{args = [200 : i32, 2000 : i32, 0 : index, 1 : index, 1 : i32, 200 : i32, 2 : index, 3 : index, 200 : i32, 1 : i32]}> : (f64, !emitc.ptr<f64>, f64, !emitc.ptr<f64>) -> ()
          %25 = expression %24 : (!emitc.ptr<f64>) -> i64 {
            %28 = call_opaque "ulmBLAS::potrf<std::ptrdiff_t>"(%24) <{args = [#emitc.opaque<"true">, 200 : i32, 0 : index, 200 : i32, 1 : i32]}> : (!emitc.ptr<f64>) -> i64
            yield %28 : i64
          }
          %26 = expression %4, %25 : (i64, i64) -> i64 {
            %28 = "emitc.constant"() <{value = 2 : i64}> : () -> i64
            %29 = cmp ne, %25, %4 : (i64, i64) -> i1
            %30 = conditional %29, %28, %4 : i64
            yield %30 : i64
          }
          %27 = expression %26, %4 : (i64, i64) -> i1 {
            %28 = cmp eq, %26, %4 : (i64, i64) -> i1
            yield %28 : i1
          }
          if %27 {
            call_opaque "ulmBLAS::trlsv<std::ptrdiff_t>"(%arg1, %arg2) <{args = [2000 : i32, #emitc.opaque<"false">, 0 : index, 2000 : i32, 1 : i32, 1 : index, 1 : i32]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
            call_opaque "ulmBLAS::gemv<std::ptrdiff_t>"(%2, %arg0, %arg2, %3, %arg3) <{args = [200 : i32, 2000 : i32, 0 : index, 1 : index, 1 : i32, 200 : i32, 2 : index, 1 : i32, 3 : index, 4 : index, 1 : i32]}> : (f64, !emitc.ptr<f64>, !emitc.ptr<f64>, f64, !emitc.ptr<f64>) -> ()
            call_opaque "ulmBLAS::trlsv<std::ptrdiff_t>"(%24, %arg3) <{args = [200 : i32, #emitc.opaque<"false">, 0 : index, 200 : i32, 1 : i32, 1 : index, 1 : i32]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
            call_opaque "ulmBLAS::trusv<std::ptrdiff_t>"(%24, %arg3) <{args = [200 : i32, #emitc.opaque<"false">, 0 : index, 1 : i32, 200 : i32, 1 : index, 1 : i32]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
          }
          assign %26 : i64 to %21 : <i64>
        } else {
          assign %19 : i64 to %21 : <i64>
        }
        %22 = expression %21 : (!emitc.lvalue<i64>) -> i64 {
          %23 = load %21 : <i64>
          yield %23 : i64
        }
        if %13 {
          call_opaque "free"(%15)  : (!emitc.ptr<i8>) -> ()
        }
        assign %22 : i64 to %11 : <i64>
      }
      %12 = expression %11 : (!emitc.lvalue<i64>) -> i64 {
        %13 = load %11 : <i64>
        yield %13 : i64
      }
      assign %12 : i64 to %8 : <i64>
    }
    %9 = expression %8 : (!emitc.lvalue<i64>) -> i64 {
      %10 = load %8 : <i64>
      yield %10 : i64
    }
    return %9 : i64
  }
  emitc.verbatim "/* gls_keep: matrices by reference (row-major), owned by the caller\0A *   v1  double[2000][200]   in    read only\0A *   v2  double[2000][2000]  in    read only\0A *   v3  double[2000][1]     in    read only\0A *   v4  double[200][1]      out   written\0A *   v5  int8_t *            work  workspace (bytes)\0A *   v6  int64_t *           info  -1: query, 0: allocate, >0: size of work\0A * returns info:\0A *     0  success\0A *     1  Cholesky factorisation of v2: not positive definite\0A *     2  Cholesky factorisation of a computed matrix: not positive definite\0A *    -6  v6: workspace too small\0A * (info != 0: the outputs are unspecified)\0A */"
  emitc.func @gls_keep(%arg0: !emitc.ptr<f64> {ulm_lam.in}, %arg1: !emitc.ptr<f64> {ulm_lam.in}, %arg2: !emitc.ptr<f64> {ulm_lam.in}, %arg3: !emitc.ptr<f64> {bufferize.result}, %arg4: !emitc.ptr<i8> {ulm_lam.workspace}, %arg5: !emitc.ptr<i64> {ulm_lam.workspace}) -> i64 attributes {specifiers = ["extern \22C\22"], ulm_lam.info = ["Cholesky factorisation of v2: not positive definite", "Cholesky factorisation of a computed matrix: not positive definite"], ulm_lam.wraps = "gls"} {
    %0 = expression  : () -> i64 {
      %18 = "emitc.constant"() <{value = -6 : i64}> : () -> i64
      yield %18 : i64
    }
    %1 = expression  : () -> i64 {
      %18 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
      yield %18 : i64
    }
    %2 = expression  : () -> i64 {
      %18 = "emitc.constant"() <{value = -1 : i64}> : () -> i64
      yield %18 : i64
    }
    %3 = expression  : () -> !emitc.size_t {
      %18 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
      yield %18 : !emitc.size_t
    }
    %4 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> i64 {
      %18 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
      %19 = load %18 : <i64>
      yield %19 : i64
    }
    %5 = expression %3 : (!emitc.size_t) -> i64 {
      %18 = cast %3 : !emitc.size_t to i64
      yield %18 : i64
    }
    %6 = expression %arg4, %5 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
      %18 = add %arg4, %5 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
      %19 = cast %18 : !emitc.ptr<i8> to !emitc.ptr<f64>
      yield %19 : !emitc.ptr<f64>
    }
    %7 = expression  : () -> i64 {
      %18 = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
      %19 = cast %18 : !emitc.size_t to i64
      yield %19 : i64
    }
    %8 = expression %arg4, %7 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
      %18 = add %arg4, %7 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
      %19 = cast %18 : !emitc.ptr<i8> to !emitc.ptr<f64>
      yield %19 : !emitc.ptr<f64>
    }
    %9 = expression  : () -> i64 {
      %18 = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
      %19 = cast %18 : !emitc.size_t to i64
      yield %19 : i64
    }
    %10 = expression %arg4, %9 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
      %18 = add %arg4, %9 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
      %19 = cast %18 : !emitc.ptr<i8> to !emitc.ptr<f64>
      yield %19 : !emitc.ptr<f64>
    }
    %11 = expression  : () -> i64 {
      %18 = "emitc.constant"() <{value = 3 : index}> : () -> !emitc.size_t
      %19 = cast %18 : !emitc.size_t to i64
      yield %19 : i64
    }
    %12 = expression %arg4, %11 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
      %18 = add %arg4, %11 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
      %19 = cast %18 : !emitc.ptr<i8> to !emitc.ptr<f64>
      yield %19 : !emitc.ptr<f64>
    }
    %13 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64> {
      %18 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
      yield %18 : !emitc.lvalue<i64>
    }
    assign %2 : i64 to %13 : <i64>
    %14 = expression %6, %8, %10, %12, %arg4, %arg5 : (!emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<i8>, !emitc.ptr<i64>) -> i64 {
      %18 = call @gls(%6, %8, %10, %12, %arg4, %arg5) {ulm_lam.query} : (!emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<i8>, !emitc.ptr<i64>) -> i64
      yield %18 : i64
    }
    %15 = expression %14, %1 : (i64, i64) -> i1 {
      %18 = cmp eq, %14, %1 : (i64, i64) -> i1
      yield %18 : i1
    }
    %16 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i64>
    if %15 {
      %18 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> i64 {
        %32 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
        %33 = load %32 : <i64>
        yield %33 : i64
      }
      %19 = expression %18 : (i64) -> ui64 {
        %32 = cast %18 : i64 to ui64
        yield %32 : ui64
      }
      %20 = expression  : () -> ui64 {
        %32 = "emitc.constant"() <{value = 63 : i64}> : () -> i64
        %33 = cast %32 : i64 to ui64
        yield %33 : ui64
      }
      %21 = expression %19, %20 : (ui64, ui64) -> i64 {
        %32 = add %19, %20 : (ui64, ui64) -> ui64
        %33 = cast %32 : ui64 to i64
        yield %33 : i64
      }
      %22 = expression %21 : (i64) -> ui64 {
        %32 = cast %21 : i64 to ui64
        yield %32 : ui64
      }
      %23 = expression  : () -> ui64 {
        %32 = "emitc.constant"() <{value = -64 : i64}> : () -> i64
        %33 = cast %32 : i64 to ui64
        yield %33 : ui64
      }
      %24 = expression %22, %23 : (ui64, ui64) -> i64 {
        %32 = bitwise_and %22, %23 : (ui64, ui64) -> ui64
        %33 = cast %32 : ui64 to i64
        yield %33 : i64
      }
      %25 = expression %24 : (i64) -> ui64 {
        %32 = cast %24 : i64 to ui64
        yield %32 : ui64
      }
      %26 = expression  : () -> ui64 {
        %32 = "emitc.constant"() <{value = 35216000 : i64}> : () -> i64
        %33 = cast %32 : i64 to ui64
        yield %33 : ui64
      }
      %27 = expression %25, %26 : (ui64, ui64) -> i64 {
        %32 = add %25, %26 : (ui64, ui64) -> ui64
        %33 = cast %32 : ui64 to i64
        yield %33 : i64
      }
      %28 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64> {
        %32 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
        yield %32 : !emitc.lvalue<i64>
      }
      assign %4 : i64 to %28 : <i64>
      %29 = expression %4, %2 : (i64, i64) -> i1 {
        %32 = cmp eq, %4, %2 : (i64, i64) -> i1
        yield %32 : i1
      }
      %30 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i64>
      if %29 {
        %32 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64> {
          %33 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
          yield %33 : !emitc.lvalue<i64>
        }
        assign %27 : i64 to %32 : <i64>
        assign %1 : i64 to %30 : <i64>
      } else {
        %32 = expression %4, %1, %27 : (i64, i64, i64) -> i1 {
          %35 = cmp lt, %4, %27 : (i64, i64) -> i1
          %36 = cmp gt, %4, %1 : (i64, i64) -> i1
          %37 = bitwise_and %36, %35 : (i1, i1) -> i1
          yield %37 : i1
        }
        %33 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i64>
        if %32 {
          assign %0 : i64 to %33 : <i64>
        } else {
          %35 = expression %4, %1 : (i64, i64) -> i1 {
            %48 = cmp eq, %4, %1 : (i64, i64) -> i1
            yield %48 : i1
          }
          %36 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<i8>>
          if %35 {
            %48 = expression %27 : (i64) -> !emitc.ptrdiff_t {
              %52 = cast %27 : i64 to !emitc.ptrdiff_t
              yield %52 : !emitc.ptrdiff_t
            }
            %49 = expression %48 : (!emitc.ptrdiff_t) -> !emitc.size_t {
              %52 = cast %48 : !emitc.ptrdiff_t to !emitc.size_t
              yield %52 : !emitc.size_t
            }
            %50 = expression %49 : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">> {
              %52 = call_opaque "aligned_alloc"(%49) <{args = [64, 0 : index]}> : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
              yield %52 : !emitc.ptr<!emitc.opaque<"void">>
            }
            %51 = expression %50 : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<i8> {
              %52 = cast %50 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i8>
              yield %52 : !emitc.ptr<i8>
            }
            assign %51 : !emitc.ptr<i8> to %36 : <!emitc.ptr<i8>>
          } else {
            assign %arg4 : !emitc.ptr<i8> to %36 : <!emitc.ptr<i8>>
          }
          %37 = expression %36 : (!emitc.lvalue<!emitc.ptr<i8>>) -> !emitc.ptr<i8> {
            %48 = load %36 : <!emitc.ptr<i8>>
            yield %48 : !emitc.ptr<i8>
          }
          %38 = expression %3 : (!emitc.size_t) -> i64 {
            %48 = cast %3 : !emitc.size_t to i64
            yield %48 : i64
          }
          %39 = expression %37, %38 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
            %48 = add %37, %38 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
            %49 = cast %48 : !emitc.ptr<i8> to !emitc.ptr<f64>
            yield %49 : !emitc.ptr<f64>
          }
          call_opaque "ulm_lam_copy"(%39, %arg0) <{args = [0 : index, 1 : index, 3200000]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
          %40 = expression  : () -> i64 {
            %48 = "emitc.constant"() <{value = 3200000 : index}> : () -> !emitc.size_t
            %49 = cast %48 : !emitc.size_t to i64
            yield %49 : i64
          }
          %41 = expression %37, %40 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
            %48 = add %37, %40 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
            %49 = cast %48 : !emitc.ptr<i8> to !emitc.ptr<f64>
            yield %49 : !emitc.ptr<f64>
          }
          call_opaque "ulm_lam_copy"(%41, %arg1) <{args = [0 : index, 1 : index, 32000000]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
          %42 = expression  : () -> i64 {
            %48 = "emitc.constant"() <{value = 35200000 : index}> : () -> !emitc.size_t
            %49 = cast %48 : !emitc.size_t to i64
            yield %49 : i64
          }
          %43 = expression %37, %42 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<f64> {
            %48 = add %37, %42 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
            %49 = cast %48 : !emitc.ptr<i8> to !emitc.ptr<f64>
            yield %49 : !emitc.ptr<f64>
          }
          call_opaque "ulm_lam_copy"(%43, %arg2) <{args = [0 : index, 1 : index, 16000]}> : (!emitc.ptr<f64>, !emitc.ptr<f64>) -> ()
          %44 = expression  : () -> i64 {
            %48 = "emitc.constant"() <{value = 35216000 : index}> : () -> !emitc.size_t
            %49 = cast %48 : !emitc.size_t to i64
            yield %49 : i64
          }
          %45 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64> {
            %48 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
            yield %48 : !emitc.lvalue<i64>
          }
          assign %18 : i64 to %45 : <i64>
          %46 = expression %39, %41, %43, %arg3, %arg5, %37, %44 : (!emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<i64>, !emitc.ptr<i8>, i64) -> i64 {
            %48 = add %37, %44 : (!emitc.ptr<i8>, i64) -> !emitc.ptr<i8>
            %49 = call @gls(%39, %41, %43, %arg3, %48, %arg5) : (!emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<f64>, !emitc.ptr<i8>, !emitc.ptr<i64>) -> i64
            yield %49 : i64
          }
          %47 = expression %46, %1 : (i64, i64) -> i1 {
            %48 = cmp eq, %46, %1 : (i64, i64) -> i1
            yield %48 : i1
          }
          if %47 {
            %48 = expression %arg5, %3 : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64> {
              %49 = subscript %arg5[%3] : (!emitc.ptr<i64>, !emitc.size_t) -> !emitc.lvalue<i64>
              yield %49 : !emitc.lvalue<i64>
            }
            assign %4 : i64 to %48 : <i64>
          }
          if %35 {
            call_opaque "free"(%37)  : (!emitc.ptr<i8>) -> ()
          }
          assign %46 : i64 to %33 : <i64>
        }
        %34 = expression %33 : (!emitc.lvalue<i64>) -> i64 {
          %35 = load %33 : <i64>
          yield %35 : i64
        }
        assign %34 : i64 to %30 : <i64>
      }
      %31 = expression %30 : (!emitc.lvalue<i64>) -> i64 {
        %32 = load %30 : <i64>
        yield %32 : i64
      }
      assign %31 : i64 to %16 : <i64>
    } else {
      assign %14 : i64 to %16 : <i64>
    }
    %17 = expression %16 : (!emitc.lvalue<i64>) -> i64 {
      %18 = load %16 : <i64>
      yield %18 : i64
    }
    return %17 : i64
  }
}

