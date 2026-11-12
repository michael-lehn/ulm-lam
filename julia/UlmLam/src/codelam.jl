# codelam.jl: @code_lam, the derivation ulm.lam found, as BLAS/LAPACK calls.
#
#     julia> @code_lam gls(X, M, y)
#     gls(X::n×m full_rank, M::n×n spd, y::n)
#       potrf    L1 ← chol(M)                      M = L1 L1ᵀ, Cholesky
#       copy     T1 ← X
#       trsm     T1 ← L1⁻¹ T1
#       ...
#
# It reads the kernels after selection (ulm_lamk on tensors, before
# bufferization) and writes each as an assignment in place: B ← L⁻¹ B
# overwrites B, as the BLAS call does.
#
# Julia matrices are column-major, the generated code is row-major: a Julia
# matrix X is, in the IR, the matrix Xᵀ (the same memory).  Each IR value is
# shown as a name and whether the IR holds that matrix or its transpose; the
# kernels are written in Julia's terms (C ← A B or, for the transpose,
# C ← Bᵀ Aᵀ).  Where the IR leaves the choice (a new temporary), the
# rendering with the fewest transposes is shown.

"""The derivation of a ulm.lam function (show it in the REPL)."""
struct CodeLam
    text::String
end
Base.show(io::IO, ::MIME"text/plain", c::CodeLam) = print(io, c.text)
Base.show(io::IO, c::CodeLam) = print(io, c.text)

# the operands of ins(...), outs(...), alpha(...), beta(...)
function operands(line, key)
    m = match(Regex(key * raw"\(([^:)]*)"), line)
    m === nothing ? String[] : [strip(x) for x in split(m.captures[1], ",") if !isempty(strip(x))]
end
attr(line, a) = (m = match(Regex(a * raw"\((\w+)\)"), line); m === nothing ? "" : m.captures[1])
defined(line) = (m = match(r"^(%[\w#]+)(?::\d+)?(?:, %\w+)* = ", line); m === nothing ? nothing : m.captures[1])

const KERNEL = r"^(%[\w#]+)(?::\d+)?(?:, %\w+)* = ulm_lamk\.(\w+)"

"""The IR lines of the function, and the places where the rendering may
choose whether a new temporary is shown as itself or as its transpose: a
kernel writing a new matrix (tensor.empty), and an scf.if with its yields."""
function choices(body)
    empties = Set{String}()
    choice = Dict{Int,Int}()      # line -> choice variable (a line)
    stack = Tuple{Symbol,Int,Int}[]  # (:if or :for, line, start of branch)
    for (k, line) in enumerate(body)
        v = defined(line)
        occursin("tensor.empty", line) && v !== nothing && push!(empties, v)
        if (m = match(KERNEL, line)) !== nothing
            outs = operands(line, "outs")
            ismat = !occursin(r"-> tensor<\?x1xf64>|-> tensor<\d+x1xf64>", line)
            if !isempty(outs) && outs[1] in empties && ismat && m.captures[2] != "potrf" &&
               m.captures[2] != "getrf" && m.captures[2] != "syrk"
                choice[k] = k
            end
        end
        if occursin(r"scf\.if", line)
            push!(stack, (:if, k, k))
            occursin(r"-> \(tensor<\?x\?xf64>|-> \(tensor<\d+x\d+xf64>", line) && (choice[k] = k)
        elseif occursin(r"scf\.for", line)
            push!(stack, (:for, k, k))
        elseif startswith(line, "} else {")
            kind, at, _ = stack[end]
            stack[end] = (kind, at, k)
        elseif line == "}" && !isempty(stack)
            pop!(stack)
        elseif startswith(line, "scf.yield") && !isempty(stack) && stack[end][1] == :if
            _, at, from = stack[end]
            haskey(choice, at) || continue
            for y in [strip(split(x, ":")[1]) for x in split(split(line, " : ")[1][10:end], ",")]
                for j in k-1:-1:from
                    if defined(body[j]) == y
                        choice[j] = at   # the if's result is this value
                        break
                    end
                end
            end
        end
    end
    choice
end

mutable struct Renderer
    vals::Dict{String,Tuple{String,Bool}}  # IR value -> (name, IR holds its transpose)
    consts::Dict{String,String}
    counter::Dict{String,Int}
    lines::Vector{String}
    params::Set{String}
    flags::Dict{Int,Bool}      # choice variable -> transposed
    choice::Dict{Int,Int}
    ifname::Dict{Int,String}   # an scf.if -> the name of its result
    sym::Set{String}           # names of symmetric temporaries (no ᵀ shown)
    inits::Set{String}         # initial values of loops
end

fresh!(r::Renderer, c) = (r.counter[c] = get(r.counter, c, 0) + 1; "$c$(r.counter[c])")
val(r::Renderer, v) = get!(() -> (fresh!(r, "T"), false), r.vals, v)
name(r, v) = val(r, v)[1]
flag(r, v) = val(r, v)[2]
# the IR value v, optionally transposed (t), in Julia's terms
disp(r, v, t::Bool=false) = (nf = val(r, v); nf[1] * ((nf[2] ⊻ t) && !(nf[1] in r.sym) ? "ᵀ" : ""))
# out as the operand of a kernel whose result is shown with orientation f
oper(r, out, f::Bool) = disp(r, out, f)
dispinv(r, v, t::Bool) = (nf = val(r, v); nf[1] * ((nf[2] ⊻ t) ? "⁻ᵀ" : "⁻¹"))

scal(r, v) = get(r.consts, v, haskey(r.vals, v) ? name(r, v) : v)
function coef(r, v)
    c = scal(r, v)
    c == "1" ? "" : c == "-1" ? "-" : c * " "
end

function emit!(r::Renderer, indent, kernel, formula, note="")
    push!(r.lines, rstrip(indent * rpad(kernel, 9) * rpad(formula, 34) * note))
end

"""The result of a kernel writing into out: in place (out's name), a copy of
a parameter (a new name), or a new temporary (a choice of orientation)."""
function result!(r::Renderer, k, res, out; prefix="T")
    if haskey(r.choice, k)
        c = r.choice[k]
        n = c == k ? fresh!(r, prefix) : r.ifname[c]
        r.vals[res] = (n, r.flags[c])
    elseif out in r.params || prefix != "T"
        r.vals[res] = (fresh!(r, prefix), flag(r, out))
    else
        r.vals[res] = val(r, out)
    end
    r.vals[res]
end

# β C + α P as text: C + P, C − P, b C + a P
function accumulate_text(β, c, αc, prod)
    β == "0" && return αc * prod
    β == "-1" && !startswith(αc, "-") && return αc * prod * " − " * c
    c = (β == "1" ? "" : β * " ") * c
    startswith(αc, "-") ? c * " − " * αc[2:end] * prod : c * " + " * αc * prod
end

function render_once(f::LamFunction, body, choice, flags, infos)
    r = Renderer(Dict(), Dict(), Dict(), String[], Set(), flags, choice, Dict(), Set(), Set())
    for line in body, m in eachmatch(r"%\w+ = (%[\w#]+)", something(match(r"iter_args\((.*)\) ->", line), (captures=[""],)).captures[1])
        push!(r.inits, m.captures[1])
    end
    head = body[1]
    args = [m.captures[1] for m in eachmatch(r"(%arg\d+):", head)]
    # a Julia matrix is its transpose in the IR; a symmetric one is itself
    sym(t) = any(p -> p in (:symmetric, :spd, :spsd), t.props)
    labels = vcat([(string(x), false) for x in f.sizes],
                  [(string(n), t isa MatType && !t.isvec && !sym(t)) for (n, t) in f.params])
    for (a, l) in zip(args, labels)
        r.vals[a] = l
        push!(r.params, a)
    end
    sig = join(["$n::" * (t isa MatType ? (t.isvec ? "$(t.rows)" : "$(t.rows)×$(t.cols)") *
                (isempty(t.props) ? "" : " " * join(t.props, ",")) : t isa FloatType ? "Float" : "Index")
                for (n, t) in f.params], ", ")
    push!(r.lines, "$(f.name)($sig)")
    ind = "  "
    depth = 1
    for (k, line) in enumerate(body)
        k == 1 && continue
        (line == "}" && depth == 1) && break
        d = ind^depth
        if (m = match(r"^(%[\w#]+) = arith.constant ([-\d.e+]+)", line)) !== nothing
            v = parse(Float64, m.captures[2])
            r.consts[m.captures[1]] = isinteger(v) ? string(Int(v)) : string(v)
            continue
        end
        if (m = match(r"^(%[\w#]+) = arith\.(\w+) (%[\w#]+), (%[\w#]+)", line)) !== nothing
            op = Dict("divf" => "/", "mulf" => "*", "addf" => "+", "subf" => "-")
            if haskey(op, m.captures[2])
                x, y = scal(r, m.captures[3]), scal(r, m.captures[4])
                r.consts[m.captures[1]] =
                    m.captures[2] == "mulf" && y == "-1" ? "-" * x :
                    m.captures[2] == "mulf" && x == "-1" ? "-" * y :
                    "(" * x * " " * op[m.captures[2]] * " " * y * ")"
            end
            continue
        end
        if (m = match(r"^(?:(%[\w#:]+) = )?scf\.for (%\w+) = \S+ to (%[\w#]+) step \S+(?: iter_args\((.*)\) ->.*)? \{", line)) !== nothing
            r.vals[m.captures[2]] = ("i", false)
            if m.captures[4] !== nothing
                res = split(m.captures[1], ":")[1]
                for (j, pair) in enumerate(split(m.captures[4], ", "))
                    a, init = strip.(split(pair, " = "))
                    n0, f0 = val(r, init)
                    # the state carried by the loop: x for the parameter x0
                    # (a copy of it, or the initial value's own buffer)
                    n = init in r.params && endswith(n0, "0") && length(n0) > 1 ? n0[1:end-1] :
                        init in r.params ? fresh!(r, "T") : n0
                    init in r.params && emit!(r, d, "copy", "$n ← $n0", "the loop's state, in place")
                    r.vals[a] = (n, f0)
                    r.vals["$res#$(j-1)"] = (n, f0)
                    j == 1 && (r.vals[res] = (n, f0))
                end
            end
            push!(r.lines, d * "for i = 0:" * name(r, m.captures[3]) * "-1")
            depth += 1
            continue
        end
        if occursin(r"scf\.if", line)
            if haskey(choice, k)
                r.ifname[k] = fresh!(r, "T")
                v = defined(line)
                v === nothing || (r.vals[v] = (r.ifname[k], flags[k]))
            end
            push!(r.lines, d * "if cheaper   (the sizes are known at run time: both plans are generated)")
            depth += 1
            continue
        end
        if startswith(line, "} else {")
            push!(r.lines, ind^(depth - 1) * "else")
            continue
        end
        if line == "}"
            depth -= 1
            continue
        end
        startswith(line, "scf.yield") && continue
        if (m = match(r"^(?:(%[\w#]+) = )?func\.call @(\w+)\(([^)]*)\)", line)) !== nothing
            sizes = Set(string.(f.sizes))
            args_ = [disp(r, strip(a)) for a in split(m.captures[3], ",") if !isempty(strip(a))]
            args_ = filter(a -> !(a in sizes), args_)
            if m.captures[1] !== nothing
                v = fresh!(r, "z")
                # a matrix from Julia is its transpose in the IR
                r.vals[m.captures[1]] = (v, !occursin(r"-> tensor<[^>]*x1xf64>\s*$", line))
                emit!(r, d, "call", "$v ← $(m.captures[2])($(join(args_, ", ")))", "your Julia function")
            else
                emit!(r, d, "call", "$(m.captures[2])($(join(args_, ", ")))", "your Julia function")
            end
            continue
        end
        if (m = match(r"^(%[\w#]+) = bufferization\.materialize_in_destination (%[\w#]+) in (?:writable )?(%[\w#]+)", line)) !== nothing
            src, dst = m.captures[2], m.captures[3]
            r.vals[m.captures[1]] = val(r, dst)
            s_ = disp(r, src, flag(r, dst))
            s_ == name(r, dst) || emit!(r, d, "copy", "$(name(r, dst)) ← $s_")
            continue
        end
        if startswith(line, "return")
            vs = [strip(v) for v in split(split(replace(line, "return" => ""), " : ")[1], ",") if !isempty(strip(v))]
            types = split(split(line, " : ")[end], ", ")
            out = map(zip(vs, types)) do (v, t)
                vec = occursin(r"x1xf64", t)
                vec ? name(r, v) : disp(r, v, true)  # Julia gets the IR's transpose
            end
            push!(r.lines, ind * "return " * join(out, ", "))
            continue
        end
        (m = match(KERNEL, line)) === nothing && continue
        res, op = m.captures
        ins, outs = operands(line, "ins"), operands(line, "outs")
        out = isempty(outs) ? "" : outs[1]
        T(x) = x == "trans"
        if op == "potrf" || op == "getrf"
            # the factors of A_ir = Aᵀ (a Julia matrix) are shown as those of A
            a = name(r, out)
            fa = flag(r, out)
            l, _ = result!(r, k, res, out; prefix=op == "potrf" ? "L" : "LU")
            r.vals[res] = r.vals["$res#0"] = (l, op == "getrf" ? fa : false)
            note = op == "getrf" ? "P $a = L U, LU" :
                   attr(line, "uplo") == "lower" ? "$a = $l $(l)ᵀ, Cholesky" : "$a = $(l)ᵀ $l, Cholesky"
            emit!(r, d, op, "$l ← $(op == "potrf" ? "chol" : "lu")($a)", note)
        elseif op == "gemm" || op == "gemv"
            a, b = ins
            ta = T(attr(line, op == "gemm" ? "trans_a" : "trans"))
            tb = op == "gemm" && T(attr(line, "trans_b"))
            β = scal(r, operands(line, "beta")[1])
            ov = β == "0" ? nothing : val(r, out)
            c, fc = result!(r, k, res, out)
            prod = fc ? disp(r, b, !tb) * " " * disp(r, a, !ta) : disp(r, a, ta) * " " * disp(r, b, tb)
            olds = ov === nothing ? "" : ov[1] * ((ov[2] ⊻ fc) && !(ov[1] in r.sym) ? "ᵀ" : "")
            rhs = accumulate_text(β, olds,
                                  coef(r, operands(line, "alpha")[1]), prod)
            emit!(r, d, op, "$c ← $rhs")
        elseif op == "syrk"
            a = ins[1]
            t = T(attr(line, "trans"))
            β = scal(r, operands(line, "beta")[1])
            olds = β == "0" ? "" : name(r, out)
            c, fc = result!(r, k, res, out)
            push!(r.sym, c)
            prod = disp(r, a, t) * " " * disp(r, a, !t)
            rhs = accumulate_text(β, olds,
                                  coef(r, operands(line, "alpha")[1]), prod)
            emit!(r, d, "syrk", "$c ← $rhs", "one triangle only")
        elseif op == "symm" || op == "symv"
            a, b = ins
            tb = T(attr(line, "trans_b"))
            c, fc = result!(r, k, res, out)
            left = attr(line, "side") != "right"
            sa = name(r, a)
            rhs = (left ⊻ fc) ? "$sa $(disp(r, b, tb ⊻ fc))" : "$(disp(r, b, tb ⊻ fc)) $sa"
            emit!(r, d, op, "$c ← $(coef(r, operands(line, "alpha")[1]))$rhs", "$sa symmetric")
        elseif op == "trsm" || op == "trsv" || op == "getrs"
            a = ins[1]
            t = T(attr(line, "trans"))
            ov = val(r, out)
            b, fb = result!(r, k, res, out)
            old = ov[1] * ((ov[2] ⊻ fb) && !(ov[1] in r.sym) ? "ᵀ" : "")
            al = op == "trsm" ? coef(r, operands(line, "alpha")[1]) : ""
            left = attr(line, "side") != "right"
            # the IR solves B_ir ← op(A)⁻¹ B_ir; with B_ir = Bᵀ that is B ← B op(A)⁻ᵀ
            inv = dispinv(r, a, t ⊻ fb)
            rhs = (left ⊻ fb) ? "$al$inv $old" : "$al$old $inv"
            emit!(r, d, op, "$b ← $rhs")
        elseif op == "gecopy"
            src = ins[1]
            b, fb = result!(r, k, res, out)
            # the initial value of a loop copied from the parameter W0: W
            if res in r.inits && src in r.params && endswith(name(r, src), "0")
                b = name(r, src)[1:end-1]
                r.vals[res] = (b, fb)
                emit!(r, d, "copy", "$b ← $(disp(r, src, T(attr(line, "trans")) ⊻ fb))",
                      "the loop's state, in place")
            else
                emit!(r, d, "copy", "$b ← $(disp(r, src, T(attr(line, "trans")) ⊻ fb))")
            end
        elseif op == "adddiag"
            old = name(r, out)
            a, _ = result!(r, k, res, out)
            old in r.sym && push!(r.sym, a)
            emit!(r, d, "adddiag", "$a ← $old + $(coef(r, operands(line, "alpha")[1]))I")
        elseif op == "gescal"
            old = name(r, out)
            a, _ = result!(r, k, res, out)
            emit!(r, d, "scal", "$a ← $(coef(r, operands(line, "alpha")[1]))$old")
        elseif op == "geaxpy"
            olds = name(r, out)
            y, fy = result!(r, k, res, out)
            emit!(r, d, "axpy", accumulate_text("1", "$y ← $olds", coef(r, operands(line, "alpha")[1]),
                                                disp(r, ins[1], fy)))
        elseif op == "sycomplete"
            old = name(r, out)
            a, _ = result!(r, k, res, out)
            push!(r.sym, a)
            emit!(r, d, "symm.", "$a ← sym($old)", "copy one triangle to the other")
        else
            a, _ = result!(r, k, res, out)
            emit!(r, d, op, "$a ← $op(" * join([disp(r, x) for x in ins], ", ") * ")")
        end
    end
    checks = sort([k for k in keys(infos) if k > 0])
    if !isempty(checks)
        push!(r.lines, "info (as LAPACK's; InfoError in Julia):")
        cname(m) = (k = parse(Int, m[2:end]); k <= length(labels) ? labels[k][1] : m)
        for k in checks
            push!(r.lines, "  " * lpad(k, 3) * "  " * replace(infos[k], r"\bv\d+" => cname))
        end
    end
    r.lines
end

"""Number the temporaries T, factors L, LU and callback results z in the
order they appear."""
function renumber(text)
    seen = Dict{String,String}()
    count = Dict{String,Int}()
    replace(text, r"\b(LU|T|L|z)(\d+)(?![0-9A-Za-z_])" => function (m)
        get!(seen, m) do
            p = match(r"^[A-Za-z]+", m).match
            count[p] = get(count, p, 0) + 1
            "$p$(count[p])"
        end
    end)
end

function render(f::LamFunction, selected::String, infos::Dict{Int,String})
    i = findfirst("func.func @$(f.name)(", selected)
    i === nothing && throw(LamError("no function $(f.name) in the IR"))
    body = [strip(l) for l in split(selected[first(i):end], '\n')]
    choice = choices(body)
    vars = sort(unique(values(choice)))
    best, score = String[], typemax(Int)
    # every orientation of the new temporaries (few), the fewest transposes
    n = length(vars) <= 12 ? length(vars) : 0
    for bits in 0:(2^n-1)
        flags = Dict(v => (j <= n && (bits >> (j - 1)) & 1 == 1) for (j, v) in enumerate(vars))
        lines = render_once(f, body, choice, flags, infos)
        s = sum(l -> count(==('ᵀ'), l), lines)
        s < score && ((best, score) = (lines, s))
    end
    renumber(join(best, "\n"))
end

"""
    code_lam(f; backend=nothing) -> CodeLam

The derivation ulm.lam found for the ulm.lam function f: its BLAS and LAPACK
calls in order, loops, decisions taken at run time, and what info means.
"""
function code_lam(f::LamFunction; backend::Union{Symbol,Nothing}=nothing)
    lib = backend === nothing ? library(f) : compile(f; name="code_lam_$(f.name)", backend=backend)
    file = replace(lib.path, r"^(.*)/lib([^/]*)\.[^.]+$" => s"\1/\2.selected.mlir")
    CodeLam(render(f, read(file, String), lib.functions[f.name].codes))
end

"""
    @code_lam f(args...)

Show the BLAS and LAPACK calls ulm.lam derived for the ulm.lam function f
(the arguments are not needed; they document the call).
"""
macro code_lam(ex)
    (ex isa Expr && ex.head == :call) || error("@code_lam needs a call: @code_lam f(x, y)")
    :(code_lam($(esc(ex.args[1]))))
end
