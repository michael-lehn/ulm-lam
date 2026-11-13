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
    lastuse::Dict{String,Int}  # the last line that uses a value
    selected::String           # the whole module (callees are defined in it)
    inout::Set{String}         # parameters the function may overwrite
    updated::Set{String}       # parameters it updates in place (update!)
    jnames::Dict{String,String}  # C name -> Julia name (foo_ -> foo!)
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
    elseif (out in r.params && !(out in r.updated)) || prefix != "T" ||
           get(r.lastuse, out, 0) > k
        # a parameter, a factor, or a value still used later: a new name
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

"""(result, argument) for the results of the function with this signature
that are the new values of arguments ({ulm_lam.inout = k}), and the number
of results."""
function updates_of(sig)
    i = findfirst(") -> (", sig)
    i === nothing && return Tuple{Int,Int}[], 0
    pieces = split(sig[first(i):end], "tensor<")[2:end]
    ups = Tuple{Int,Int}[]
    for (j, p) in enumerate(pieces)
        m = match(r"ulm_lam\.inout = (\d+)", p)
        m === nothing || push!(ups, (j - 1, parse(Int, m.captures[1])))
    end
    ups, length(pieces)
end

function render_once(f::LamFunction, body, choice, flags, infos, selected="",
                     jnames=Dict{String,String}())
    r = Renderer(Dict(), Dict(), Dict(), String[], Set(), flags, choice, Dict(), Set(), Set(),
                 Dict{String,Int}(), selected, Set{String}(), Set{String}(), jnames)
    jname(c) = get(r.jnames, c, c)
    for m in eachmatch(r"(%arg\d+): [^{,]*\{([^}]*)\}", body[1])
        occursin("ulm_lam.inout", m.captures[2]) && push!(r.inout, m.captures[1])
    end
    for (k, line) in enumerate(body), m in eachmatch(r"(%[\w#]+)", line)
        d = defined(line)
        m.captures[1] == d && continue
        r.lastuse[m.captures[1]] = k
    end
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
    # parameters it updates (LAPACK style: y holds the solution on exit)
    ups, _ = updates_of(head)
    for (_, a) in ups
        a < length(args) && push!(r.updated, args[a+1])
    end
    sig = join(["$n::" * (t isa MatType ? (t.isvec ? "$(t.rows)" : "$(t.rows)×$(t.cols)") *
                (isempty(t.props) ? "" : " " * join(t.props, ",")) *
                (t.intent == :inout ? " inout" : "") : t isa FloatType ? "Float" : "Index")
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
        if (m = match(r"^(?:(%[\w#]+)(?::\d+)? = )?(?:func\.)?call @(\w+)\(([^)]*)\)", line)) !== nothing
            # a ulm.lam function (defined in the module) or a callback
            isdef = occursin("func.func @$(m.captures[2])(", r.selected)
            note = isdef ? "a ulm.lam function" : "your Julia function"
            # an inout parameter of the callee gets a copy of a value the
            # caller must not overwrite (a parameter of its own, or a value
            # it still uses): bufferization inserts that copy
            ops_ = [strip(x) for x in split(m.captures[3], ",") if !isempty(strip(x))]
            cups, nres = Tuple{Int,Int}[], 0
            if isdef
                sig = r.selected[first(findfirst("func.func @$(m.captures[2])(", r.selected)):end]
                sig = sig[1:first(findfirst("{\n", sig))]
                cups, nres = updates_of(sig)
                cargs = [something(c.captures[1], "") for c in
                         eachmatch(r"%arg\d+: [^{,]*(?:\{([^}]*)\})?", sig)]
                for (j, x) in enumerate(ops_)
                    if j <= length(cargs) && occursin("ulm_lam.inout", cargs[j]) &&
                       ((x in r.params && !(x in r.inout)) || get(r.lastuse, x, 0) > k)
                        c_ = fresh!(r, "T")
                        emit!(r, d, "copy", "$c_ ← $(disp(r, x))", "$(jname(m.captures[2])) may overwrite it")
                        r.vals[x * "@copy"] = (c_, flag(r, x))
                        ops_[j] = x * "@copy"
                    end
                end
            end
            sizes = Set(string.(f.sizes))
            args_ = [disp(r, a) for a in ops_]
            args_ = filter(a -> !(a in sizes), args_)
            callee = jname(m.captures[2])
            if !isempty(cups)
                # the new values of the arguments it updates are those
                # arguments (in place); the other results are new
                upd = Dict(cups)
                own, changed = String[], String[]
                for j in 0:nres-1
                    key = nres == 1 ? m.captures[1] : "$(m.captures[1])#$j"
                    if haskey(upd, j)
                        r.vals[key] = val(r, ops_[upd[j]+1])
                        push!(changed, name(r, key))
                    else
                        r.vals[key] = (fresh!(r, "z"), true)
                        push!(own, name(r, key))
                    end
                end
                lhs = isempty(own) ? "" : join(own, ", ") * " ← "
                emit!(r, d, "call", "$lhs$callee($(join(args_, ", ")))",
                      "$note, overwrites $(join(changed, ", "))")
            elseif m.captures[1] !== nothing
                v = fresh!(r, "z")
                # a matrix from Julia is its transpose in the IR
                r.vals[m.captures[1]] = (v, !occursin(r"-> tensor<[^>]*x1xf64>\s*$", line))
                emit!(r, d, "call", "$v ← $callee($(join(args_, ", ")))", note)
            else
                emit!(r, d, "call", "$callee($(join(args_, ", ")))", note)
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
            # the new values of updated parameters: in place, or copied
            upd = Dict(ups)
            for (j, v) in enumerate(vs)
                haskey(upd, j - 1) || continue
                p_ = args[upd[j-1]+1]
                name(r, v) == name(r, p_) ||
                    emit!(r, ind, "copy", "$(name(r, p_)) ← $(disp(r, v, flag(r, p_)))")
            end
            out = [occursin(r"x1xf64", t) ? name(r, v) : disp(r, v, true)  # Julia gets the IR's transpose
                   for (j, (v, t)) in enumerate(zip(vs, types)) if !haskey(upd, j - 1)]
            (isempty(out) && !isempty(ups)) || push!(r.lines, ind * "return " * join(out, ", "))
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
            sym = op == "getrf" && occursin("symmetric", line)  # L D Lᵀ
            l, _ = result!(r, k, res, out; prefix=op == "potrf" ? "L" : sym ? "LDL" : "LU")
            r.vals[res] = r.vals["$res#0"] = (l, op == "getrf" && !sym ? fa : false)
            note = sym ? "$a = L D Lᵀ, Bunch–Kaufman" : op == "getrf" ? "P $a = L U, LU" :
                   attr(line, "uplo") == "lower" ? "$a = $l $(l)ᵀ, Cholesky" : "$a = $(l)ᵀ $l, Cholesky"
            emit!(r, d, sym ? "sytrf" : op,
                  "$l ← $(op == "potrf" ? "chol" : sym ? "ldlt" : "lu")($a)", note)
        elseif op == "gels"
            a, bb = outs
            la, lb = disp(r, a), disp(r, bb)
            r.vals[res] = r.vals["$res#0"] = val(r, a)
            r.vals["$res#1"] = val(r, bb)
            emit!(r, d, "gels", "$lb ← lstsq($la, $lb)", "QR; the solution in the leading rows")
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
        elseif op == "gemmtr"
            a, bb = ins
            ta, tb = T(attr(line, "trans_a")), T(attr(line, "trans_b"))
            β = scal(r, operands(line, "beta")[1])
            olds = β == "0" ? "" : name(r, out)
            c, fc = result!(r, k, res, out)
            push!(r.sym, c)
            prod = fc ? disp(r, bb, !tb) * " " * disp(r, a, !ta) : disp(r, a, ta) * " " * disp(r, bb, tb)
            rhs = accumulate_text(β, olds, coef(r, operands(line, "alpha")[1]), prod)
            emit!(r, d, "gemmtr", "$c ← $rhs", "symmetric: one triangle only")
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
            emit!(r, d, op == "getrs" && occursin("symmetric", line) ? "sytrs" : op, "$b ← $rhs")
        elseif op == "trmm"
            a = ins[1]
            t = T(attr(line, "trans"))
            ov = val(r, out)
            b, fb = result!(r, k, res, out)
            old = ov[1] * ((ov[2] ⊻ fb) && !(ov[1] in r.sym) ? "ᵀ" : "")
            al = coef(r, operands(line, "alpha")[1])
            left = attr(line, "side") != "right"
            # B_ir ← op(A) B_ir; with B_ir = Bᵀ that is B ← B op(A)ᵀ
            fa = disp(r, a, t ⊻ fb)
            emit!(r, d, "trmm", "$b ← $al" * ((left ⊻ fb) ? "$fa $old" : "$old $fa"),
                  "$(name(r, a)) triangular")
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
        elseif op == "potri" || op == "getri"
            f = name(r, out)
            x, _ = result!(r, k, res, out; prefix="X")
            if op == "potri"
                emit!(r, d, op, "$x ← ($f $(f)ᵀ)⁻¹", "explicit inverse, one triangle")
            else
                emit!(r, d, op, "$x ← inv($f)", "explicit inverse from the LU factors")
            end
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
        # not errors: the checks at run time and what a failure reports
        push!(r.lines, "if a factorisation fails, the call throws InfoError with")
        cname(m) = (k = parse(Int, m[2:end]); k <= length(labels) ? labels[k][1] : m)
        for k in checks
            push!(r.lines, "  info $k:  " * replace(infos[k], r"\bv\d+" => cname))
        end
    end
    r.lines
end

"""Number the temporaries T, factors L, LU and callback results z in the
order they appear."""
function renumber(text)
    seen = Dict{String,String}()
    count = Dict{String,Int}()
    replace(text, r"\b(LDL|LU|T|L|z)(\d+)(?![0-9A-Za-z_])" => function (m)
        get!(seen, m) do
            p = match(r"^[A-Za-z]+", m).match
            count[p] = get(count, p, 0) + 1
            "$p$(count[p])"
        end
    end)
end

function render(f::LamFunction, selected::String, infos::Dict{Int,String},
                jnames=Dict{String,String}())
    i = findfirst("func.func @$(cname(f))(", selected)
    i === nothing && throw(LamError("no function $(f.name) in the IR"))
    body = [strip(l) for l in split(selected[first(i):end], '\n')]
    # (only this function: the next one uses the same names again)
    nxt = findnext(l -> startswith(l, "func.func"), body, 2)
    nxt === nothing || (body = body[1:nxt-1])
    choice = choices(body)
    vars = sort(unique(values(choice)))
    best, score = String[], typemax(Int)
    # every orientation of the new temporaries (few), the fewest transposes
    n = length(vars) <= 12 ? length(vars) : 0
    for bits in 0:(2^n-1)
        flags = Dict(v => (j <= n && (bits >> (j - 1)) & 1 == 1) for (j, v) in enumerate(vars))
        lines = render_once(f, body, choice, flags, infos, selected, jnames)
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
    lib = backend === nothing ? library(f) : compile(f; name="code_lam_$(cname(f))", backend=backend)
    file = replace(lib.path, r"^(.*)/lib([^/]*)\.[^.]+$" => s"\1/\2.selected.mlir")
    jnames = Dict(cname(c.fn) => string(c.fn.name) for c in values(lib.functions))
    CodeLam(render(f, read(file, String), lib.functions[f.name].codes, jnames))
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
