# Vector spaces R^n and R^(m x n): sums, differences, negation, scalars
# (constants and run-time Float) on either side, division by scalars,
# linear combinations; with transposes, products, solves and explicit
# inverses.  Every expression against Julia; all in one library.
using UlmLAM, LinearAlgebra, Random, Test

@testset "vector spaces" begin
    Random.seed!(3)
    n, k = 6, 4
    x, y, z = randn(n), randn(n), randn(n)
    A, B, C = randn(n, n), randn(n, n), randn(n, n)
    G, H = randn(n, k), randn(n, k)
    S0 = randn(n, n); S = S0 * S0' + n * I
    a, b = 1.7, -0.6
    exprs = [
     # vectors
     "x + y", "x - y", "-x", "x + y + z", "x - y + z", "(x + y) + z", "x + (y + z)", "x + y - x",
     "2x", "x * 2", "x / 2", "2 \\ x", "-2x", "2 * (x + y)", "(x + y) * 2", "(x + y) / 2", "2x + 3y",
     "2x - 3y", "(2 + 3) * x", "2 * (3 * x)", "-(x + y)", "-(-x)", "x - x", "0 * x", "1 * x",
     "a * x", "x * a", "x / a", "a * x + b * y", "a * (x + y)", "(a + b) * x", "a * b * x", "-a * x", "x - a * y",
     # matrices
     "A + B", "A - B", "-A", "2A", "A * 2", "A / 2", "2A + 3B", "a * A + b * B", "a * (A + B)", "A + B + C",
     "A' + B", "(A + B)'", "2 * A'", "A + 2I", "A - I", "I - A", "a * I + A", "-(A + B)",
     # products in combinations
     "A * x + y", "A * x - y", "y - A * x", "2 * (A * x)", "(A * x) * 2", "(A * x) / 2", "a * A * x + b * y",
     "A * x + B * y", "A * (x + y)", "(A + B) * x", "(A - B) * x", "A * B + C", "2 * A * B - C", "(A + B) * C",
     "A * (B + C)", "A * B * x", "(A * B) * 2", "a * (A * B)", "A * B * a", "-(A * B)", "A * B - B * A",
     # transposes and inner/outer products
     "A' * x", "x' * y", "x * y'", "x' * A * y", "G' * x", "G * G'", "G' * G", "A' * A",
     # inverses and solves combined with vector-space ops
     "S \\ x", "S \\ (x + y)", "S \\ x + y", "2 * (S \\ x)", "(S \\ x) * 2", "(S \\ x) / a", "inv(S) * x - y",
     "inv(2S) * x", "inv(S + I) * x", "inv(a * S) * x", "x - S * (S \\ x)", "A \\ x", "A \\ (2x)", "(A + I) \\ x",
     "inv(A) * B", "inv(A)", "inv(S)", "S \\ A + B", "inv(A') * x", "A' \\ x",
     "inv(A) + B", "2 * inv(S)", "inv(S) - I", "x / a", "a \\ x", "(x + y) / a", "A / a",
     # rectangular matrices: R^(n x k)
     "G + H", "G - H", "-G", "2G", "G * 2", "G / 2", "a * G", "G / a", "a * G + b * H", "a * (G + H)",
     "(a + b) * G", "G + H - G", "0 * G", "1 * G", "G' + H'", "(G + H)'", "G' - H'", "2 * G'",
     "(G + H)' * x", "G' * x + G' * y", "A * G + H", "A * (G + H)", "(A + B) * G", "G * G' + A",
     "S \\ G", "S \\ (G + H)", "G' * (S \\ G)", "G' * x", "(G - H)' * (x - y)",
     # a run-time scalar defined after the product it scales
     "a * (A * x) + B * x - inv(A) * x / a", "inv(S) * x / a + y",
    ]
    types = Dict("x" => :(Vec(:n)), "y" => :(Vec(:n)), "z" => :(Vec(:n)),
                 "A" => :(Mat(:n, :n)), "B" => :(Mat(:n, :n)), "C" => :(Mat(:n, :n)),
                 "G" => :(Mat(:n, :k)), "H" => :(Mat(:n, :k)), "S" => :(Mat(:n, :n, :spd)),
                 "a" => :Float, "b" => :Float)
    vals = Dict("x" => x, "y" => y, "z" => z, "A" => A, "B" => B, "C" => C, "G" => G,
                "H" => H, "S" => S, "a" => a, "b" => b)
    names_in(ex) = ex isa Symbol ? Set([string(ex)]) :
                   ex isa Expr ? union(Set{String}(), map(names_in, ex.args)...) : Set{String}()
    fs, used = UlmLAM.LamFunction[], Vector{String}[]
    for (i, e) in enumerate(exprs)
        u = [v for v in ("x", "y", "z", "A", "B", "C", "G", "H", "S", "a", "b")
             if v in names_in(Meta.parse(e))]
        args = join(["$v::$(types[v])" for v in u], ", ")
        push!(fs, Core.eval(@__MODULE__, Meta.parse("@lam vs$i($args) = $e")))
        push!(used, u)
    end
    lib = UlmLAM.compile(fs; name="vector_spaces")
    for (i, e) in enumerate(exprs)
        args = [vals[v] for v in used[i]]
        jf = Core.eval(@__MODULE__, Meta.parse("(" * join(used[i], ", ") *
                                                (length(used[i]) == 1 ? "," : "") * ") -> $e"))
        ref = Base.invokelatest(jf, args...)
        r = getproperty(lib, Symbol("vs$i"))(args...)
        # (a scalar such as x' * y is a vector of length 1 in ulm.lam)
        ref isa Number && (ref = [ref])
        ok = size(r) == size(ref) && norm(r - ref) <= 1e-10 * max(norm(ref), 1)
        ok || @info "vector spaces: $e"
        @test ok
    end
end
