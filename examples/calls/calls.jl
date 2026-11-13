# foo calls bar, and bar needs a workspace (julia --project=julia/UlmLam
# examples/calls/calls.jl; the same in Python: calls.py)
using UlmLam, LinearAlgebra, Random

@lam bar(M::Mat(:n, :n, :spd), y::Vec(:n)) = M \ y
@lam foo(M::Mat(:n, :n, :spd), N::Mat(:n, :n, :spd), y::Vec(:n)) = bar(M, bar(N, y))

Random.seed!(1)
n = 5
A, B = randn(n, n), randn(n, n)
M, N, y = A * A' + n * I, B * B' + n * I, randn(n)

x = foo(M, N, y)                     # compiled at the first call, bar with it
lib = UlmLam.library(foo)            # the library: foo and bar
println("error: ", norm(x - M \ (N \ y)) / norm(x))
println("workspace of bar: ", workspace_size(lib.bar, M, y), " bytes, of foo: ",
        workspace_size(lib.foo, M, N, y), " bytes")
# the IR as traced, and the generated C++
write(joinpath(@__DIR__, "generated", "calls_jl.mlir"), first(UlmLam.lam_module([foo])))
write(joinpath(@__DIR__, "generated", "calls_jl.cpp"), lib.source)
display(@code_lam foo(M, N, y)); println()
display(@code_lam bar(M, y)); println()

# The same with bar allowed to overwrite M (InOut): bar needs no workspace;
# foo2, whose M and N are only read, passes copies (in its own workspace);
# foo3, which may overwrite M and N itself, passes them on.
@lam bar2(M::InOut(Mat(:n, :n, :spd)), y::Vec(:n)) = M \ y
@lam foo2(M::Mat(:n, :n, :spd), N::Mat(:n, :n, :spd), y::Vec(:n)) = bar2(M, bar2(N, y))
@lam foo3(M::InOut(Mat(:n, :n, :spd)), N::InOut(Mat(:n, :n, :spd)), y::Vec(:n)) =
    bar2(M, bar2(N, y))

lib2 = UlmLam.compile([foo2, foo3]; name="calls_inout_jl")
M2, N2 = copy(M), copy(N)
x2 = lib2.foo2(M2, N2, y)
M3, N3 = copy(M), copy(N)
x3 = lib2.foo3(M3, N3, y)
println("errors: ", norm(x2 - M \ (N \ y)) / norm(x2), ", ", norm(x3 - M \ (N \ y)) / norm(x3),
        "; M after foo2 unchanged: ", M2 == M, ", after foo3 overwritten: ", M3 != M)
println("workspace of bar2: ", workspace_size(lib2.bar2, M, y), " bytes, of foo2: ",
        workspace_size(lib2.foo2, M, N, y), " bytes, of foo3: ",
        workspace_size(lib2.foo3, M, N, y), " bytes")
write(joinpath(@__DIR__, "generated", "calls_inout_jl.mlir"), first(UlmLam.lam_module([foo2, foo3])))
write(joinpath(@__DIR__, "generated", "calls_inout_jl.cpp"), lib2.source)
display(code_lam(foo2)); println()
display(code_lam(foo3)); println()
display(code_lam(bar2)); println()

# LAPACK style: bar! and foo! overwrite y with the solution (and M, N with
# their Cholesky factors), so neither needs a workspace; foo_matlab is the
# user-friendly wrapper that leaves its arguments alone: the copies, and so
# the workspace, are in it only.  (foo! is foo_ in the IR and in C.)
@lam bar!(M::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) = update!(y, M \ y)
@lam foo!(M::InOut(Mat(:n, :n, :spd)), N::InOut(Mat(:n, :n, :spd)), y::InOut(Vec(:n))) =
    update!(y, bar!(M, bar!(N, y)))
@lam foo_matlab(M::Mat(:n, :n, :spd), N::Mat(:n, :n, :spd), y::Vec(:n)) = foo!(M, N, y)

lib3 = UlmLam.compile([foo_matlab]; name="calls_lapack_jl")   # and foo!, bar!
M4, N4, y4 = copy(M), copy(N), copy(y)
lib3.foo!(M4, N4, y4)                # y4 is the solution now
x5 = lib3.foo_matlab(M, N, y)
println("errors: ", norm(y4 - M \ (N \ y)) / norm(y4), ", ", norm(x5 - M \ (N \ y)) / norm(x5),
        "; after foo!: upper triangle of M ≈ cholesky(M).U: ",
        triu(M4) ≈ cholesky(Symmetric(M)).U)
println("workspace (LAPACK style) of bar: ", workspace_size(lib3.bar!, M, y), " bytes, of foo: ",
        workspace_size(lib3.foo!, M, N, y), " bytes; of foo_matlab: ",
        workspace_size(lib3.foo_matlab, M, N, y), " bytes")
write(joinpath(@__DIR__, "generated", "calls_lapack_jl.mlir"), first(UlmLam.lam_module([foo_matlab])))
write(joinpath(@__DIR__, "generated", "calls_lapack_jl.cpp"), lib3.source)
display(code_lam(foo!)); println()
display(code_lam(bar!)); println()
display(code_lam(foo_matlab)); println()
