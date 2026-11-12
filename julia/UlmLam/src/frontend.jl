# frontend.jl: write linear algebra in Julia, get ulm_lam IR.
#
#     @lam function gls(X::Mat(:n, :m, :full_rank), M::Mat(:n, :n, :spd),
#                       y::Vec(:n))
#         inv(X' * inv(M) * X) * X' * inv(M) * y
#     end
#
# Calling the function with ulm_lam values traces it: * is ulm_lam.mul (or
# ulm_lam.scale with a number or a Float parameter), + and - are
# ulm_lam.add, X' and transpose(X) are ulm_lam.transpose, inv(X) is
# ulm_lam.inverse, A \ B is inv(A) * B and B / A is B * inv(A); I (from
# LinearAlgebra) and eye(n) are the identity, which is never formed.
#
# Julia arrays are column-major, the ulm.lam ABI is row-major: a Julia
# matrix X (rows x cols) is, as memory, the row-major matrix X^T (cols x
# rows).  So a matrix parameter has the transposed type in the IR and the
# traced function sees transpose(P); a matrix result is returned
# transposed.  Kernel selection resolves these transposes with strides:
# no matrix is copied for it.  Vectors (Vec) are the same in both orders.

const PROPERTIES = (:square, :symmetric, :spsd, :spd, :lower_tri,
                    :upper_tri, :unit_diag, :identity, :zero, :orthogonal,
                    :permutation, :non_singular, :full_rank, :diagonal)

"""An error in a ulm.lam program (shapes, scopes, signatures)."""
struct LamError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LamError) = print(io, "LamError: ", e.msg)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

const Dim = Union{Int,Symbol}

"""A matrix type: sizes (numbers or names of run-time sizes), properties,
intent (:in or :inout), and whether it is a Julia vector."""
struct MatType
    rows::Dim
    cols::Dim
    props::Vector{Symbol}  # sorted, unique
    intent::Symbol
    isvec::Bool
end

struct IndexType end
struct FloatType end
"""An index parameter (a loop trip count, an argument of an external
function)."""
const Index = IndexType()
"""A scalar parameter known only at run time (e.g. a noise variance)."""
const Float = FloatType()

const LamType = Union{MatType,IndexType,FloatType}

"""
    Mat(rows, cols, props...; stored=nothing)

A matrix of Float64 with properties, e.g. `Mat(:n, :n, :spd)`.  Sizes are
numbers, or symbols that are run-time sizes (taken from the arrays; equal
symbols are equal sizes).  `stored=:lower` or `:upper` (symmetric matrices
only): only that triangle is referenced.
"""
function Mat(rows::Dim, cols::Dim, props::Symbol...; stored=nothing)
    for p in props
        p in PROPERTIES || throw(LamError("unknown matrix property :$p; " *
                                          "known: $(join(PROPERTIES, ", "))"))
    end
    for d in (rows, cols)
        d isa Int && d <= 0 && throw(LamError("matrix sizes must be " *
                                              "positive, got $rows x $cols"))
    end
    ps = Set(props)
    if stored !== nothing
        stored in (:lower, :upper) ||
            throw(LamError("stored must be :lower or :upper"))
        isempty(intersect(ps, (:symmetric, :spsd, :spd, :identity))) &&
            throw(LamError("stored= needs a symmetric matrix"))
        push!(ps, Symbol(stored, "_stored"))
    end
    MatType(rows, cols, sort!(collect(ps)), :in, false)
end

"""`Vec(n, props...)`: a vector (a Julia `Vector{Float64}`), n x 1."""
Vec(n::Dim, props::Symbol...) =
    (t = Mat(n, 1, props...); MatType(t.rows, 1, t.props, :in, true))

"""`InOut(Mat(...))`: a matrix parameter that the function may overwrite."""
InOut(t::MatType) = MatType(t.rows, t.cols, t.props, :inout, t.isvec)

plain(t::MatType) = MatType(t.rows, t.cols, Symbol[], :in, t.isvec)
symbols(t::MatType) = [d for d in (t.rows, t.cols) if d isa Symbol]
symbols(::LamType) = Symbol[]

const FLIP = Dict(:lower_tri => :upper_tri, :upper_tri => :lower_tri,
                  :lower_stored => :upper_stored,
                  :upper_stored => :lower_stored)

"""The type of the row-major IR matrix that holds a Julia matrix of type t:
t^T (triangles flipped), a vector as it is."""
function abitype(t::MatType)
    t.isvec && return t
    MatType(t.cols, t.rows, sort!([get(FLIP, p, p) for p in t.props]),
            t.intent, false)
end

function mlir(t::MatType; props=true)
    d(x) = x isa Symbol ? "?" : string(x)
    s = "tensor<$(d(t.rows))x$(d(t.cols))xf64"
    if props && !isempty(t.props)
        s *= ", #ulm_lam.props<" * join(string.(t.props), ", ") * ">"
    end
    s * ">"
end
mlir(::IndexType; props=true) = "index"
mlir(::FloatType; props=true) = "f64"

function dims_attr(t::MatType, pos::Dict{Symbol,Int})
    s = symbols(t)
    isempty(s) ? "" : "ulm_lam.dims = [" * join([pos[x] for x in s], ", ") * "]"
end

Base.show(io::IO, t::MatType) =
    print(io, t.rows, "x", t.cols, isempty(t.props) ? "" : " " * join(t.props, ","))

# ---------------------------------------------------------------------------
# Builder: textual IR with nested regions
# ---------------------------------------------------------------------------

const REGION_IDS = Ref(0)

mutable struct Region
    id::Int
    lines::Vector{String}
    indent::String
    constants::Dict{Tuple{String,String},Any}
end
Region(indent) = (REGION_IDS[] += 1;
                  Region(REGION_IDS[], String[], indent, Dict()))

mutable struct Builder
    counter::Int
    regions::Vector{Region}
end
Builder() = Builder(0, [Region("  ")])

const CURRENT = Ref{Union{Builder,Nothing}}(nothing)

function builder()
    CURRENT[] === nothing &&
        throw(LamError("ulm_lam values exist only while a ulm.lam function " *
                       "is traced"))
    CURRENT[]
end
fresh!(b::Builder) = (b.counter += 1; "%v$(b.counter)")
region(b::Builder) = b.regions[end]
emit!(b::Builder, line) = push!(region(b).lines, region(b).indent * line)
live(b::Builder, id) = any(r -> r.id == id, b.regions)

# ---------------------------------------------------------------------------
# Values and operations
# ---------------------------------------------------------------------------

"""A value of a traced ulm.lam function (a matrix, an index or a scalar)."""
struct LamValue
    name::String
    type::LamType
    region::Int
end

function use(v::LamValue)
    live(builder(), v.region) ||
        throw(LamError("$(v.name) ($(v.type)) is used outside the loop " *
                       "that defines it"))
    v.name
end

function mat(v::LamValue, what)
    v.type isa MatType || throw(LamError("$what needs a matrix, got $(v.type)"))
    v.type
end

function constant!(b::Builder, value, t::LamType)
    key = (repr(value), string(t))
    for r in reverse(b.regions)
        haskey(r.constants, key) && return r.constants[key]
    end
    v = LamValue(fresh!(b), t, region(b).id)
    if t isa IndexType
        emit!(b, "$(v.name) = arith.constant $(Int(value)) : index")
    else
        emit!(b, "$(v.name) = arith.constant $(repr(Float64(value))) : f64")
    end
    region(b).constants[key] = v
    v
end

function op!(name, operands::Vector{LamValue}, result::MatType)
    b = builder()
    names = join(use.(operands), ", ")
    types = join([mlir(o.type) for o in operands], ", ")
    v = LamValue(fresh!(b), result, region(b).id)
    emit!(b, "$(v.name) = $name $names : $types -> $(mlir(result))")
    v
end

newtype(r, c) = MatType(r, c, Symbol[], :in, false)

function Base.transpose(a::LamValue)
    t = mat(a, "transpose")
    op!("ulm_lam.transpose", [a], newtype(t.cols, t.rows))
end
Base.adjoint(a::LamValue) = transpose(a)  # real matrices

function Base.inv(a::LamValue)
    t = mat(a, "inv")
    t.rows == t.cols || throw(LamError("inverse of a non-square $t matrix"))
    op!("ulm_lam.inverse", [a], newtype(t.rows, t.cols))
end

function Base.:*(a::LamValue, b::LamValue)
    a.type isa FloatType && b.type isa FloatType && return scalar_op("mulf", a, b)
    a.type isa FloatType && return scale_by(a, b)
    b.type isa FloatType && return scale_by(b, a)
    ta, tb = mat(a, "*"), mat(b, "*")
    ta.cols == tb.rows ||
        throw(LamError("product of $ta and $tb: $(ta.cols) columns but " *
                       "$(tb.rows) rows"))
    op!("ulm_lam.mul", [a, b], newtype(ta.rows, tb.cols))
end
Base.:*(alpha::Real, a::LamValue) =
    a.type isa FloatType ? scalar_op("mulf", alpha, a) : scale(Float64(alpha), a)
Base.:*(a::LamValue, alpha::Real) = alpha * a

# scalar arithmetic of Float values (1 / beta, alpha / beta, ...)
function Base.:/(a::LamValue, b::LamValue)
    a.type isa FloatType && b.type isa FloatType && return scalar_op("divf", a, b)
    b.type isa FloatType && return scalar_op("divf", 1.0, b) * a  # X / s
    a * inv(b)
end
Base.:/(a::Real, b::LamValue) = scalar_op("divf", a, b)
Base.:/(a::LamValue, b::Real) =
    a.type isa FloatType ? scalar_op("divf", a, b) : (1.0 / b) * a
Base.:+(a::Real, b::LamValue) = scalar_op("addf", a, b)
Base.:+(a::LamValue, b::Real) = scalar_op("addf", a, b)
Base.:-(a::Real, b::LamValue) = scalar_op("subf", a, b)
Base.:-(a::LamValue, b::Real) = scalar_op("subf", a, b)

"""x op y for Float values and numbers (arith.addf, subf, mulf, divf)."""
function scalar_op(op, x, y)
    b = builder()
    vals = map((x, y)) do a
        a isa Real && (a = constant!(b, Float64(a), Float))
        (a isa LamValue && a.type isa FloatType) ||
            throw(LamError("$op: scalars only, got $a"))
        a
    end
    v = LamValue(fresh!(b), Float, region(b).id)
    emit!(b, "$(v.name) = arith.$op $(use(vals[1])), $(use(vals[2])) : f64")
    v
end
function Base.:\(a::LamValue, b::LamValue)
    a.type isa FloatType && return b.type isa FloatType ? scalar_op("divf", b, a) :
                                   scalar_op("divf", 1.0, a) * b
    t = mat(a, "\\")
    t.rows == t.cols && return inv(a) * b
    # a rectangular A: least squares, as Julia's A \ B
    lstsq(a, b)
end

"""
    lstsq(A, B)

The X that minimises ||A X - B|| (A m x n with m >= n and full column rank),
as `A \\ B` for a rectangular A: QR (LAPACK's gels) where the backend has
LAPACK, otherwise the normal equations with a Cholesky factorisation.
"""
function lstsq(a::LamValue, b::LamValue)
    ta, tb = mat(a, "lstsq"), mat(b, "lstsq")
    ta.rows == tb.rows ||
        throw(LamError("lstsq: A has $(ta.rows) rows, B has $(tb.rows)"))
    op!("ulm_lam.lstsq", [a, b], newtype(ta.cols, tb.cols))
end
Base.:\(a::Real, b::LamValue) = b / a

# A^k: products (k > 0), the identity (k = 0), inverses (k < 0)
function Base.:^(a::LamValue, k::Integer)
    t = mat(a, "^")
    t.rows == t.cols || throw(LamError("A^k needs a square matrix, got $(t.rows)×$(t.cols)"))
    k == 0 && return eye(t.rows)
    k < 0 && return inv(a)^(-k)
    k == 1 ? a : a * a^(k - 1)
end
LinearAlgebra.dot(a::LamValue, b::LamValue) = transpose(a) * b
# common functions that are not linear algebra ulm.lam maps (yet): a clear error
for (m, f) in ((:LinearAlgebra, :tr), (:LinearAlgebra, :det), (:LinearAlgebra, :logdet),
               (:LinearAlgebra, :norm), (:LinearAlgebra, :diag), (:Base, :sum),
               (:Base, :exp), (:Base, :log), (:Base, :sqrt), (:Base, :abs), (:Base, :size))
    @eval $m.$f(::LamValue, args...) =
        throw(LamError(string($(QuoteNode(f)), " is not supported in a @lam function " *
                              "(supported: *, +, -, ', transpose, inv, \\, /, ^, dot, " *
                              "scalars, I, assume, loop, externs)")))
end
Base.broadcastable(::LamValue) =
    throw(LamError("elementwise operations (x .* y, f.(x)) are not supported in a " *
                   "@lam function: compute them in Julia, e.g. in a callback"))


function Base.:+(a::LamValue, b::LamValue)
    a.type isa FloatType && return scalar_op("addf", a, b)
    ta, tb = mat(a, "+"), mat(b, "+")
    (ta.rows, ta.cols) == (tb.rows, tb.cols) ||
        throw(LamError("sum of $ta and $tb: different shapes"))
    op!("ulm_lam.add", [a, b], newtype(ta.rows, ta.cols))
end
Base.:-(a::LamValue, b::LamValue) =
    a.type isa FloatType ? scalar_op("subf", a, b) : a + (-1.0) * b
Base.:-(a::LamValue) = a.type isa FloatType ? scalar_op("subf", 0.0, a) : (-1.0) * a

function scale(alpha::Float64, x::LamValue)
    t = mat(x, "scaling")
    b = builder()
    c = constant!(b, alpha, Float)
    op!("ulm_lam.scale", [c, x], newtype(t.rows, t.cols))
end

function scale_by(alpha::LamValue, x::LamValue)
    t = mat(x, "scaling by $(alpha.name)")
    op!("ulm_lam.scale", [alpha, x], newtype(t.rows, t.cols))
end

# The identity: I (LinearAlgebra.UniformScaling) and eye(n).

"""`alpha * I`, alpha a number or a Float parameter (never formed)."""
struct ScaledI
    alpha::Union{Float64,LamValue}
end
function Base.:*(s::LamValue, J::UniformScaling)
    s.type isa FloatType || return J.λ == 1 ? s : Float64(J.λ) * s  # A * I = A
    ScaledI(J.λ == 1 ? s : scalar_op("mulf", Float64(J.λ), s))
end
Base.:*(J::UniformScaling, s::LamValue) = s * J
Base.:+(a::LamValue, J::UniformScaling) = a + ScaledI(Float64(J.λ))
Base.:+(J::UniformScaling, a::LamValue) = a + J
Base.:-(a::LamValue, J::UniformScaling) = a + ScaledI(-Float64(J.λ))
Base.:-(J::UniformScaling, a::LamValue) = (-a) + J
function Base.:+(a::LamValue, s::ScaledI)
    t = mat(a, "+ I")
    t.rows == t.cols || throw(LamError("$t + I: not square"))
    e = eye(t.rows)
    a + (s.alpha isa LamValue ? s.alpha * e : s.alpha * e)
end
Base.:+(s::ScaledI, a::LamValue) = a + s
Base.:-(a::LamValue, s::ScaledI) =
    a + ScaledI(s.alpha isa LamValue ? (-1.0) * s.alpha : -s.alpha)

"""`eye(n)`: the n x n identity (n a number or a size of the function);
in a sum it is added to the diagonal, never formed."""
function eye(n::Dim)
    t = Mat(n, n, :identity)
    b = builder()
    sizes = n isa Symbol ? size_values([n, n], "eye") : String[]
    v = LamValue(fresh!(b), t, region(b).id)
    operands = isempty(sizes) ? "" : "(" * join(sizes, ", ") * ")"
    emit!(b, "$(v.name) = ulm_lam.identity$operands : $(mlir(t))")
    v
end

"""
    assume(x, props...)

`x` with properties the compiler cannot derive, e.g.
`assume(K + s2 * I, :spd)` for s2 > 0 and K positive semidefinite.  Trusted;
a factorisation that finds it wrong returns info > 0 (`InfoError`).
"""
function assume(x::LamValue, props::Symbol...)
    t = mat(x, "assume")
    r = Mat(t.rows, t.cols, props...)
    b = builder()
    v = LamValue(fresh!(b), r, region(b).id)
    emit!(b, "$(v.name) = ulm_lam.assume $(use(x)) : $(mlir(t)) -> $(mlir(r))")
    v
end

"""`x` with a type without properties (dropping them is sound)."""
function plainvalue(x::LamValue)
    t = x.type
    (t isa MatType && !isempty(t.props)) || return x
    b = builder()
    v = LamValue(fresh!(b), plain(t), region(b).id)
    emit!(b, "$(v.name) = ulm_lam.assume $(use(x)) : $(mlir(t)) -> " *
             "$(mlir(plain(t)))")
    v
end

function cast(x::LamValue, t::MatType)
    b = builder()
    r = MatType(t.rows, t.cols, t.props, :in, false)
    v = LamValue(fresh!(b), r, region(b).id)
    emit!(b, "$(v.name) = ulm_lam.assume $(use(x)) : $(mlir(x.type)) -> " *
             "$(mlir(r))")
    v
end

# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

"""
    loop(body, n, carry...)

A loop with `n` iterations (a number or an Index parameter):

    loop(steps) do i
        y = observe(i)
        record(i, inv(S) * y)
    end

With loop-carried matrices the body gets them after the index and returns
their next values; `loop` returns their final values:

    (X,) = loop(n, X0) do i, X
        (X + A * B * C,)
    end
"""
function loop(body, n, carry::LamValue...)
    b = builder()
    c0 = constant!(b, 0, Index)
    c1 = constant!(b, 1, Index)
    ub = n isa Integer ? constant!(b, n, Index) : n
    ub isa LamValue && ub.type isa IndexType ||
        throw(LamError("the trip count of a loop is an Index"))
    inits = [plainvalue(c) for c in carry]
    result = fresh!(b)
    push!(b.regions, Region(region(b).indent * "  "))
    index = LamValue(fresh!(b), Index, region(b).id)
    carried = [LamValue(fresh!(b), plain(c.type), region(b).id) for c in inits]
    local next
    try
        next = body(index, carried...)
    catch
        pop!(b.regions)
        rethrow()
    end
    body_region = pop!(b.regions)
    k = length(inits)
    if k > 0
        next isa LamValue && (next = (next,))
        (next isa Tuple && length(next) == k) ||
            throw(LamError("the loop body must return the $k carried values"))
        next = [plainvalue(v) for v in next]
        for (v, c) in zip(next, carried)
            mlir(v.type) == mlir(c.type) ||
                throw(LamError("loop-carried value: $(v.type) instead of " *
                               "$(c.type)"))
        end
    end
    head = "scf.for $(index.name) = $(c0.name) to $(use(ub)) step $(c1.name)"
    if k > 0
        iters = join(["$(c.name) = $(use(i))" for (c, i) in zip(carried, inits)], ", ")
        types = join([mlir(i.type) for i in inits], ", ")
        res = k > 1 ? "$result:$k" : result
        emit!(b, "$res = $head iter_args($iters) -> ($types) {")
    else
        emit!(b, "$head {")
    end
    append!(region(b).lines, body_region.lines)
    if k > 0
        emit!(b, "  scf.yield " * join([v.name for v in next], ", ") * " : " *
                 join([mlir(v.type) for v in next], ", "))
    end
    emit!(b, "}")
    k == 0 && return nothing
    names = k == 1 ? [result] : ["$result#$j" for j in 0:k-1]
    Tuple(LamValue(nm, i.type, region(b).id) for (nm, i) in zip(names, inits))
end

# ---------------------------------------------------------------------------
# External functions
# ---------------------------------------------------------------------------

"""
    extern(name, params, result=nothing)

An external function the program calls, implemented in Julia with
`bind!(lib; name = f)`:

    observe = extern(:observe, [Index], Vec(2000))     # y = observe(i)
    record  = extern(:record, [Index, Vec(200)])        # record(i, b)

The callback gets indices as Int and matrices as views of the memory of
the generated code (vectors as Vector, matrices as lazy transposes of the
row-major buffers); copy what it keeps.
"""
struct Extern
    name::Symbol
    params::Vector{LamType}
    result::Union{MatType,Nothing}
end
extern(name::Symbol, params, result=nothing) =
    Extern(name, collect(LamType, params), result)

function ext_sizes(e::Extern)
    out = Symbol[]
    for t in (e.params..., something(e.result, Index))
        for x in symbols(t)
            x in out || push!(out, x)
        end
    end
    out
end

function declaration(e::Extern)
    sizes = ext_sizes(e)
    pos = Dict(x => k - 1 for (k, x) in enumerate(sizes))
    params = join(vcat(fill("index", length(sizes)), [mlir(p) for p in e.params]), ", ")
    res = ""
    if e.result !== nothing
        # a matrix result is written by Julia column-major: the transposed type
        rt = e.result.isvec ? e.result : abitype(e.result)
        d = dims_attr(rt, pos)
        res = " -> (" * mlir(rt) * (isempty(d) ? "" : " {$d}") * ")"
    end
    "func.func private @$(e.name)($params)$res"
end

function (e::Extern)(args...)
    b = builder()
    length(args) == length(e.params) ||
        throw(LamError("$(e.name) takes $(length(e.params)) arguments"))
    vals = LamValue[]
    for (a, p) in zip(args, e.params)
        a isa Integer && p isa IndexType && (a = constant!(b, a, Index))
        a isa LamValue || throw(LamError("$(e.name): unexpected argument $a"))
        if p isa MatType
            (a.type isa MatType && (a.type.rows, a.type.cols) == (p.rows, p.cols)) ||
                throw(LamError("$(e.name): argument $(a.type) instead of $p"))
            mlir(a.type) == mlir(p) || (a = cast(a, p))
        elseif typeof(a.type) != typeof(p)
            throw(LamError("$(e.name): argument $(a.type) instead of $p"))
        end
        push!(vals, a)
    end
    PROGRAM[].externs[e.name] = e
    sizes = ext_sizes(e)
    names = join(vcat(size_values(sizes, string(e.name)), use.(vals)), ", ")
    ptypes = join(vcat(fill("index", length(sizes)), [mlir(p) for p in e.params]), ", ")
    if e.result === nothing
        emit!(b, "func.call @$(e.name)($names) : ($ptypes) -> ()")
        return nothing
    end
    rt = e.result.isvec ? e.result : abitype(e.result)
    v = LamValue(fresh!(b), rt, region(b).id)
    emit!(b, "$(v.name) = func.call @$(e.name)($names) : ($ptypes) -> " * mlir(rt))
    # Julia's column-major matrix is the transpose of the row-major IR value
    e.result.isvec ? v : transpose(v)
end

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

mutable struct Program
    externs::Dict{Symbol,Extern}
end
const PROGRAM = Ref{Union{Program,Nothing}}(nothing)

"""A ulm.lam function (from `@lam`)."""
mutable struct LamFunction
    name::Symbol
    params::Vector{Pair{Symbol,LamType}}
    fn::Function
    sizes::Vector{Symbol}
    results::Vector{MatType}  # Julia types of the results
    text::Union{String,Nothing}
end

function LamFunction(name::Symbol, params::Vector, fn::Function)
    ps = Pair{Symbol,LamType}[]
    for (n, t) in params
        t isa LamType || throw(LamError("$name: parameter $n: $t is not a " *
                                        "ulm.lam type (Mat, Vec, Index, Float)"))
        push!(ps, n => t)
    end
    sizes = Symbol[]
    for (_, t) in ps, x in symbols(t)
        x in sizes || push!(sizes, x)
    end
    LamFunction(name, ps, fn, sizes, MatType[], nothing)
end

Base.show(io::IO, f::LamFunction) =
    print(io, "ulm.lam function ", f.name, "(", join(["$n::$t" for (n, t) in f.params], ", "), ")")

const TRACING = Ref{Union{LamFunction,Nothing}}(nothing)

"""The values of run-time sizes in the function being traced."""
function size_values(sizes::Vector{Symbol}, callee::String)
    f = TRACING[]
    known = f === nothing ? Symbol[] : f.sizes
    map(sizes) do x
        x in known || throw(LamError("$callee needs the size :$x, which no " *
                                     "parameter of the calling function has"))
        "%s.$x"
    end
end

function trace!(f::LamFunction)
    f.text !== nothing && return f.text
    outer_b, outer_f = CURRENT[], TRACING[]
    b = Builder()
    CURRENT[], TRACING[] = b, f
    pos = Dict(x => k - 1 for (k, x) in enumerate(f.sizes))
    results = String[]
    rtypes = String[]
    try
        # the parameters as the traced function sees them (a Julia matrix is
        # the transpose of the row-major IR parameter)
        args = map(f.params) do (n, t)
            p = LamValue("%p.$n", t isa MatType ? abitype(t) : t, region(b).id)
            (t isa MatType && !t.isvec) ? transpose(p) : p
        end
        out = try
            f.fn(args...)
        catch e
            # a Julia function without a method for ulm.lam values
            if e isa MethodError && any(x -> x isa LamValue, e.args)
                throw(LamError("$(f.name): $(e.f) is not supported in a @lam function " *
                               "(supported: *, +, -, ', transpose, inv, \\, /, ^, dot, " *
                               "scalars, I, assume, loop, externs)"))
            end
            rethrow()
        end
        outs = out === nothing ? () : out isa Tuple ? out : (out,)
        f.results = MatType[]
        for o in outs
            o isa LamValue && o.type isa MatType ||
                throw(LamError("$(f.name): a result must be a matrix"))
            t = o.type
            for x in symbols(t)
                haskey(pos, x) || throw(LamError("$(f.name): the size :$x of a " *
                                                 "result is not a size of a parameter"))
            end
            # n x 1: a Julia vector; otherwise returned transposed
            isvec = t.cols == 1
            jt = MatType(t.rows, t.cols, Symbol[], :in, isvec)
            push!(f.results, jt)
            r = isvec ? plainvalue(o) : plainvalue(transpose(o))
            push!(results, use(r))
            at = abitype(jt)
            d = dims_attr(at, pos)
            push!(rtypes, mlir(r.type) * " {ulm_lam.out" * (isempty(d) ? "" : ", $d") * "}")
        end
        emit!(b, isempty(results) ? "return" :
                 "return " * join(results, ", ") * " : " *
                 join([split(t, " {")[1] for t in rtypes], ", "))
    finally
        CURRENT[], TRACING[] = outer_b, outer_f
    end
    args = ["%s.$x: index" for x in f.sizes]
    for (n, t) in f.params
        if t isa MatType
            at = abitype(t)
            a = [t.intent == :inout ? "ulm_lam.inout" : "ulm_lam.in"]
            d = dims_attr(at, pos)
            isempty(d) || push!(a, d)
            push!(args, "%p.$n: $(mlir(at)) {" * join(a, ", ") * "}")
        else
            push!(args, "%p.$n: $(mlir(t))")
        end
    end
    res = isempty(rtypes) ? "" : " -> (" * join(rtypes, ", ") * ")"
    head = "func.func @$(f.name)(" * join(args, ", ") * ")$res {"
    f.text = join(vcat([head], region(b).lines, ["}"]), "\n")
end

"""
    @lam function name(x::Mat(...), ...) ... end

A ulm.lam function: traced to ulm_lam IR, compiled with `UlmLam.compile`.
"""
macro lam(fdef)
    (fdef isa Expr && fdef.head in (:function, :(=))) ||
        error("@lam needs a function definition")
    sig, body = fdef.args
    sig.head == :call || error("@lam: unsupported signature $sig")
    name = sig.args[1]
    names = Symbol[]
    types = Any[]
    for p in sig.args[2:end]
        (p isa Expr && p.head == :(::)) ||
            error("@lam: parameter $p needs a type (Mat, Vec, Index, Float)")
        push!(names, p.args[1])
        push!(types, p.args[2])
    end
    params = Expr(:vect, [:($(QuoteNode(n)) => $(esc(t))) for (n, t) in zip(names, types)]...)
    lambda = Expr(:->, Expr(:tuple, names...), body)
    quote
        $(esc(name)) = $LamFunction($(QuoteNode(name)), $params, $(esc(lambda)))
    end
end

"""The ulm_lam IR of a module with these functions."""
function lam_module(functions::Vector{LamFunction})
    outer = PROGRAM[]
    prog = Program(Dict{Symbol,Extern}())
    PROGRAM[] = prog
    try
        for f in functions
            f.text = nothing
            trace!(f)
        end
    finally
        PROGRAM[] = outer
    end
    parts = [declaration(e) for e in values(prog.externs)]
    append!(parts, [f.text for f in functions])
    join(parts, "\n\n") * "\n", prog.externs
end
