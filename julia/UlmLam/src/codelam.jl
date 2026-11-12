# codelam.jl: @code_lam, the derivation ulm.lam found, as BLAS/LAPACK calls.
#
#     julia> @code_lam gls(X, M, y)
#     gls(X::n×m full_rank, M::n×n spd, y::n)
#       potrf   L1 ← chol(M)                   M = L1 L1ᵀ, lower
#       ...
#
# It reads the kernels after selection (ulm_lamk on tensors, before
# bufferization) and writes each as an assignment in place: B ← L⁻¹ B
# overwrites B, as the BLAS call does.  A Julia matrix X is passed as the
# row-major X' (no copy); where the kernels work on that buffer it appears
# as X'.

"""The derivation of a ulm.lam function (show it in the REPL)."""
struct CodeLam
    text::String
end
Base.show(io::IO, ::MIME"text/plain", c::CodeLam) = print(io, c.text)
Base.show(io::IO, c::CodeLam) = print(io, c.text)

const SUP = Dict('T' => "ᵀ", '-' => "⁻", '1' => "¹")

struct Renderer
    names::Dict{String,String}
    consts::Dict{String,String}
    counter::Dict{Char,Int}
    lines::Vector{String}
end

fresh!(r::Renderer, c::Char) = (r.counter[c] = get(r.counter, c, 0) + 1; "$c$(r.counter[c])")
function nm(r::Renderer, v)
    haskey(r.names, v) && return r.names[v]
    startswith(v, "%") && (r.names[v] = fresh!(r, 'T'))  # a new temporary
    get(r.names, v, v)
end

# the operands of ins(...) and outs(...)
function operands(line, key)
    m = match(Regex(key * raw"\(([^:)]*)"), line)
    m === nothing ? String[] : [strip(x) for x in split(m.captures[1], ",") if !isempty(strip(x))]
end
attr(line, a) = (m = match(Regex(a * raw"\((\w+)\)"), line); m === nothing ? "" : m.captures[1])

scal(r, v) = (c = get(r.consts, v, nothing); c === nothing ? nm(r, v) : c)
function coef(r, v)
    c = scal(r, v)
    c == "1" ? "" : c == "-1" ? "-" : c * " "
end
# X' transposed is X (a Julia matrix in its row-major buffer)
tr(s, t) = t == "trans" ? (endswith(s, "'") ? s[1:end-1] : s * "ᵀ") : s
inv_(s, t) = t == "trans" ? s * "⁻ᵀ" : s * "⁻¹"

function emitline!(r::Renderer, indent, kernel, formula, note="")
    push!(r.lines, indent * rpad(kernel, 9) * rpad(formula, 34) * note)
end

"""Name the result of a kernel: in place on a temporary, or a new name."""
function result!(r::Renderer, res, out, c::Char, params)
    if out in params || c == 'L'
        r.names[res] = fresh!(r, c)  # a factor, or a parameter is copied
    else
        r.names[res] = nm(r, out)    # in place
    end
    r.names[res]
end

# beta * C + alpha * P as text: C + P, C − P, b C + a P
function accumulate_text(r, β, out, αc, prod)
    β == "0" && return αc * prod
    c = (β == "1" ? "" : β * " ") * nm(r, out)
    startswith(αc, "-") ? c * " − " * αc[2:end] * prod : c * " + " * αc * prod
end

function render(f::LamFunction, selected::String, infos::Dict{Int,String})
    # the function in the selected IR
    i = findfirst("func.func @$(f.name)(", selected)
    i === nothing && throw(LamError("no function $(f.name) in the IR"))
    body = split(selected[first(i):end], '\n')
    r = Renderer(Dict(), Dict(), Dict(), String[])
    params = Set{String}()
    # arguments: the sizes, then the parameters
    head = body[1]
    args = [m.captures[1] for m in eachmatch(r"(%arg\d+):", head)]
    # a Julia matrix is X' in the IR; a symmetric one is its own transpose
    sym(t) = any(p -> p in (:symmetric, :spd, :spsd), t.props)
    labels = vcat([string(x) for x in f.sizes],
                  [t isa MatType && !t.isvec && !sym(t) ? "$n'" : string(n) for (n, t) in f.params])
    for (a, l) in zip(args, labels)
        r.names[a] = l
        push!(params, a)
    end
    sig = join(["$n::" * (t isa MatType ? (t.isvec ? "$(t.rows)" : "$(t.rows)×$(t.cols)") *
                (isempty(t.props) ? "" : " " * join(t.props, ",")) : t isa FloatType ? "Float" : "Index")
                for (n, t) in f.params], ", ")
    push!(r.lines, "$(f.name)($sig)")
    indent = "  "
    depth = 1
    for raw in body[2:end]
        line = strip(raw)
        (line == "}" && depth == 1) && break
        if (m = match(r"^(%[\w#]+) = arith.constant ([-\d.e+]+)", line)) !== nothing
            v = parse(Float64, m.captures[2])
            r.consts[m.captures[1]] = isinteger(v) ? string(Int(v)) : string(v)
            continue
        end
        if (m = match(r"^(%[\w#]+) = arith\.(\w+) (%[\w#]+), (%[\w#]+)", line)) !== nothing
            op = Dict("divf" => "/", "mulf" => "*", "addf" => "+", "subf" => "-")
            if haskey(op, m.captures[2])
                r.consts[m.captures[1]] = "(" * scal(r, m.captures[3]) * " " * op[m.captures[2]] *
                                          " " * scal(r, m.captures[4]) * ")"
            end
            continue
        end
        if (m = match(r"^(?:(%[\w#:]+) = )?scf\.for (%\w+) = \S+ to (%[\w#]+) step \S+(?: iter_args\((.*)\) ->.*)? \{", line)) !== nothing
            r.names[m.captures[2]] = "i"
            if m.captures[4] !== nothing
                res = split(m.captures[1], ":")[1]
                for (k, pair) in enumerate(split(m.captures[4], ", "))
                    a, init = strip.(split(pair, " = "))
                    r.names[a] = nm(r, init)
                    r.names["$res#$(k-1)"] = nm(r, init)
                    k == 1 && (r.names[res] = nm(r, init))
                end
            end
            push!(r.lines, indent^depth * "for i = 1:" * nm(r, m.captures[3]))
            depth += 1
            continue
        end
        if occursin(r"scf\.if", line)
            m = match(r"^(%[\w#]+) = scf\.if", line)
            m !== nothing && (r.names[m.captures[1]] = fresh!(r, 'T'))
            push!(r.lines, indent^depth * "if cheaper   (decided at run time from the sizes or the trip count)")
            depth += 1
            continue
        end
        if startswith(line, "} else {")
            push!(r.lines, indent^(depth - 1) * "else")
            continue
        end
        if line == "}"
            depth -= 1
            continue
        end
        if startswith(line, "scf.yield")
            continue
        end
        if (m = match(r"^(?:(%[\w#]+) = )?func\.call @(\w+)\(([^)]*)\)", line)) !== nothing
            args_ = [nm(r, strip(a)) for a in split(m.captures[3], ",") if !isempty(strip(a))]
            idx = filter(a -> a == "i" || !(a in string.(f.sizes)), args_)
            if m.captures[1] !== nothing
                v = fresh!(r, 'z')
                r.names[m.captures[1]] = v
                emitline!(r, indent^depth, "call", "$v ← $(m.captures[2])($(join(idx, ", ")))", "your Julia function")
            else
                emitline!(r, indent^depth, "call", "$(m.captures[2])($(join(idx, ", ")))", "your Julia function")
            end
            continue
        end
        if (m = match(r"^(%[\w#]+) = bufferization\.materialize_in_destination (%[\w#]+) in (?:writable )?(%[\w#]+)", line)) !== nothing
            r.names[m.captures[1]] = nm(r, m.captures[3])
            emitline!(r, indent^depth, "copy", "$(nm(r, m.captures[3])) ← $(nm(r, m.captures[2]))")
            continue
        end
        if startswith(line, "return")
            vals = [nm(r, strip(v)) for v in split(split(replace(line, "return" => ""), " : ")[1], ",")
                    if !isempty(strip(v))]
            push!(r.lines, indent * "return " * join(vals, ", "))
            continue
        end
        (m = match(r"^(%[\w#]+)(?:, %\w+)? = ulm_lamk\.(\w+)", line)) === nothing && continue
        res, op = m.captures
        ins, outs = operands(line, "ins"), operands(line, "outs")
        out = isempty(outs) ? "" : outs[1]
        d = indent^depth
        if op == "potrf"
            a = nm(r, out)
            l = result!(r, res, out, 'L', params)
            emitline!(r, d, "potrf", "$l ← chol($a)", attr(line, "uplo") == "lower" ? "$a = $l $(l)ᵀ, Cholesky" : "$a = $(l)ᵀ $l, Cholesky")
        elseif op == "getrf"
            a = nm(r, out)
            l = result!(r, res, out, 'L', params)
            emitline!(r, d, "getrf", "$l ← lu($a)", "P $a = L U, LU")
        elseif op == "gemm" || op == "gemv"
            a, b = ins
            prod = tr(nm(r, a), attr(line, op == "gemm" ? "trans_a" : "trans")) *
                   " " * (op == "gemm" ? tr(nm(r, b), attr(line, "trans_b")) : nm(r, b))
            rhs = accumulate_text(r, scal(r, operands(line, "beta")[1]), out,
                                  coef(r, operands(line, "alpha")[1]), prod)
            c = result!(r, res, out, 'T', params)
            emitline!(r, d, op, "$c ← $rhs")
        elseif op == "syrk"
            a = nm(r, ins[1])
            prod = attr(line, "trans") == "trans" ? "$(tr(a, "trans")) $a" : "$a $(tr(a, "trans"))"
            rhs = accumulate_text(r, scal(r, operands(line, "beta")[1]), out,
                                  coef(r, operands(line, "alpha")[1]), prod)
            c = result!(r, res, out, 'T', params)
            emitline!(r, d, "syrk", "$c ← $rhs", "$(attr(line, "uplo")) triangle only")
        elseif op == "symm" || op == "symv"
            a, b = ins
            c = result!(r, res, out, 'T', params)
            left = attr(line, "side") != "right"
            rhs = left ? "$(nm(r, a)) $(tr(nm(r, b), attr(line, "trans_b")))" : "$(tr(nm(r, b), attr(line, "trans_b"))) $(nm(r, a))"
            emitline!(r, d, op, "$c ← $(coef(r, operands(line, "alpha")[1]))$rhs", "$(nm(r, a)) symmetric")
        elseif op == "trsm" || op == "trsv" || op == "getrs"
            a = ins[1]
            b = result!(r, res, out, 'T', params)
            t = attr(line, "trans")
            al = op == "trsm" ? coef(r, operands(line, "alpha")[1]) : ""
            rhs = attr(line, "side") == "right" ? "$al$(nm(r, out)) $(inv_(nm(r, a), t))" : "$al$(inv_(nm(r, a), t)) $(nm(r, out))"
            emitline!(r, d, op, "$b ← $rhs")
        elseif op == "gecopy"
            b = result!(r, res, out, 'T', params)
            emitline!(r, d, "copy", "$b ← $(tr(nm(r, ins[1]), attr(line, "trans")))")
        elseif op == "adddiag"
            a = result!(r, res, out, 'T', params)
            emitline!(r, d, "adddiag", "$a ← $(nm(r, out)) + $(coef(r, operands(line, "alpha")[1]))I")
        elseif op == "gescal"
            a = result!(r, res, out, 'T', params)
            emitline!(r, d, "scal", "$a ← $(coef(r, operands(line, "alpha")[1]))$(nm(r, out))")
        elseif op == "geaxpy"
            y = result!(r, res, out, 'T', params)
            emitline!(r, d, "axpy", "$y ← $(nm(r, out)) + $(coef(r, operands(line, "alpha")[1]))$(nm(r, ins[1]))")
        elseif op == "sycomplete"
            a = result!(r, res, out, 'T', params)
            emitline!(r, d, "symm.", "$a ← sym($(nm(r, out)))", "upper triangle from the lower")
        else
            a = result!(r, res, out, 'T', params)
            emitline!(r, d, op, "$a ← $op(" * join([nm(r, x) for x in ins], ", ") * ")")
        end
    end
    checks = sort([k for k in keys(infos) if k > 0])
    if !isempty(checks)
        push!(r.lines, "info (as LAPACK's; InfoError in Julia):")
        cname(m) = (k = parse(Int, m[2:end]); k <= length(labels) ? replace(labels[k], "'" => "") : m)
        for k in checks
            push!(r.lines, "  " * lpad(k, 3) * "  " * replace(infos[k], r"\bv\d+" => cname))
        end
    end
    join(r.lines, "\n")
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
