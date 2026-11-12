//===- UlmLamWorkspace.cpp - LAPACK-style workspaces for temporaries ------===//
//
// ulm-lam-workspace: a function that needs memory gets it from its caller, in
// the style of the lwork arguments of LAPACK routines:
//
//   void f(..., int8_t *work, int64_t *workspace_info)
//
//   *workspace_info == -1  query: set *workspace_info to the required size
//                          (bytes) and return
//   *workspace_info ==  0  allocate one block of the required size, use it,
//                          free it before returning
//   *workspace_info  >  0  the size of `work` (bytes): if it is too small,
//                          return info = -i (i: the position of
//                          workspace_info); otherwise use `work`
//
// Which functions take a workspace:
//   * a defined function with temporaries (memref.alloc after
//     bufferization), and, transitively, every defined function that calls
//     a function taking a workspace;
//   * an external function declared with the unit attribute
//     `ulm_lam.workspace`: its last two parameters (added here) follow the
//     convention.
//
// Accumulation.  As a LAPACK routine that calls other routines, a function
// answers a query with its own requirement plus those of its callees: at
// entry it queries every call of a function with a workspace
// (*workspace_info = -1, matrix arguments are dummies that a query does not
// touch, scalars not available at entry are 0, the caller's
// *workspace_info is restored afterwards).  So the requirement of an
// external function with ulm_lam.workspace may depend only on its (static)
// matrix sizes and on scalar arguments that are available at entry.  The block
// is laid out as [own temporaries | callee 1 | callee 2 | ...], every part
// 64-byte aligned; each call gets its part of the block and its size in
// *workspace_info.  Own temporaries (also those in loops) become views at
// offsets computed from the sizes; their deallocations disappear.  The
// branches of an scf.if (alternative plans) share their part of the block:
// the required size is the one of the branch that runs (its condition is
// computed at entry) or, if the condition is not available at entry, the
// larger one.
// Recursive calls between functions with a workspace are rejected.
//
// "Too small" is marked here by a call of ulm_lam_workspace_error(required,
// given); ulm-lam-info replaces it by the negative info and removes the
// function.  The workspace queries of callees generated at entry carry the
// attribute ulm_lam.query (they are not numbered as checks).
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

constexpr int64_t kAlign = 64;
constexpr llvm::StringLiteral kErrorFn = "ulm_lam_workspace_error";
constexpr llvm::StringLiteral kWorkspaceAttr = "ulm_lam.workspace";

struct UlmLamWorkspace : PassWrapper<UlmLamWorkspace, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamWorkspace)

    StringRef
    getArgument() const final
    {
	return "ulm-lam-workspace";
    }
    StringRef
    getDescription() const final
    {
	return "Temporaries from a caller-provided workspace (LAPACK-style "
	       "query, allocate, or check), accumulated over calls";
    }
    void
    getDependentDialects(DialectRegistry &registry) const override
    {
	registry.insert<memref::MemRefDialect, scf::SCFDialect,
	                arith::ArithDialect, func::FuncDialect>();
    }

    MemRefType bufTy, infoTy;

    void
    runOnOperation() override
    {
	ModuleOp module = getOperation();
	MLIRContext *ctx = &getContext();
	bufTy =
	    MemRefType::get({ShapedType::kDynamic}, IntegerType::get(ctx, 8));
	infoTy = MemRefType::get({1}, IntegerType::get(ctx, 64));
	SymbolTable symbols(module);
	// sizes as function parameters, not queries of buffers
	ulm_lam::resolveDimOps(module);

	// 1. which functions take a workspace (transitively over calls)
	llvm::SetVector<func::FuncOp> ws;
	auto funcs = llvm::to_vector(module.getOps<func::FuncOp>());
	for (func::FuncOp f : funcs) {
	    if (f.isExternal()) {
		if (f->hasAttr(kWorkspaceAttr))
		    ws.insert(f);
	    } else if (!f.getOps<memref::AllocOp>().empty() ||
	               f->walk([](memref::AllocOp) {
		            return WalkResult::interrupt();
	                }).wasInterrupted()) {
		ws.insert(f);
	    }
	}
	auto callsWorkspace = [&](func::FuncOp f) {
	    bool any = false;
	    f.walk([&](func::CallOp c) {
		if (auto g = symbols.lookup<func::FuncOp>(c.getCallee()))
		    any |= ws.contains(g);
	    });
	    return any;
	};
	for (bool changed = true; changed;) {
	    changed = false;
	    for (func::FuncOp f : funcs)
		if (!f.isExternal() && !ws.contains(f) && callsWorkspace(f)) {
		    ws.insert(f);
		    changed = true;
		}
	}
	if (ws.empty())
	    return;
	for (func::FuncOp f : ws)
	    if (!f.isExternal() && f.getNumResults() != 0) {
		f.emitError("ulm-lam-workspace: function returns values (run "
		            "buffer-results-to-out-params first)");
		return signalPassFailure();
	    }

	// 2. callees before callers (and no recursion)
	SmallVector<func::FuncOp> order;
	llvm::DenseMap<Operation *, int> state; // 1 visiting, 2 done
	std::function<LogicalResult(func::FuncOp)> visit =
	    [&](func::FuncOp f) -> LogicalResult {
	    if (state[f] == 2)
		return success();
	    if (state[f] == 1)
		return f.emitError("ulm-lam-workspace: recursive calls between "
		                   "functions with a workspace");
	    state[f] = 1;
	    LogicalResult r = success();
	    f.walk([&](func::CallOp c) {
		auto g = symbols.lookup<func::FuncOp>(c.getCallee());
		if (g && ws.contains(g) && !g.isExternal() && failed(visit(g)))
		    r = failure();
	    });
	    state[f] = 2;
	    order.push_back(f);
	    return r;
	};
	for (func::FuncOp f : ws)
	    if (!f.isExternal() && failed(visit(f)))
		return signalPassFailure();

	// 3. signatures: (..., int8_t *work, int64_t *workspace_info), tagged
	//    ulm_lam.workspace so that the lowering can document them
	DictionaryAttr tag = DictionaryAttr::get(
	    ctx, {NamedAttribute(StringAttr::get(ctx, kWorkspaceAttr),
	                         UnitAttr::get(ctx))});
	for (func::FuncOp f : ws) {
	    unsigned n = f.getNumArguments();
	    if (failed(f.insertArguments({n, n}, {bufTy, infoTy}, {tag, tag},
	                                 {f.getLoc(), f.getLoc()}))) {
		f.emitError("cannot add workspace arguments");
		return signalPassFailure();
	    }
	}

	// 4. bodies
	for (func::FuncOp f : order)
	    if (failed(rewrite(f, symbols, ws)))
		return signalPassFailure();

	if (!module.lookupSymbol(kErrorFn)) {
	    OpBuilder b(module.getBodyRegion());
	    b.setInsertionPointToStart(module.getBody());
	    auto i64 = b.getI64Type();
	    auto decl = func::FuncOp::create(b, module.getLoc(), kErrorFn,
	                                     b.getFunctionType({i64, i64}, {}));
	    decl.setPrivate();
	}
    }

    /// `v` as a value available at the entry of the function: an argument
    /// or a (cloned) constant; null otherwise.
    static Value
    atEntry(OpBuilder &b, Value v, Block &entry)
    {
	if (!v)
	    return {};
	if (auto a = dyn_cast<BlockArgument>(v))
	    return a.getOwner() == &entry ? v : Value();
	Operation *def = v.getDefiningOp();
	if (def && isa<arith::ConstantOp>(def))
	    return def->getBlock() == b.getInsertionBlock() &&
	                   def->isBeforeInBlock(&*b.getInsertionPoint())
	               ? v
		       : b.clone(*def)->getResult(0);
	return {};
    }

    /// Operands for querying `call` at function entry: matrices are dummy
    /// views of `work` (a query does not touch them), scalars must be
    /// available at entry.
    FailureOr<SmallVector<Value>>
    queryOperands(OpBuilder &b, func::CallOp call, Block &entry, Value work)
    {
	SmallVector<Value> ops;
	Location loc = call.getLoc();
	Value c0;
	for (Value v : call.getOperands()) {
	    if (auto t = dyn_cast<MemRefType>(v.getType())) {
		SmallVector<Value> sizes;
		for (unsigned d = 0, r = t.getRank(); d < r; ++d) {
		    if (!t.isDynamicDim(d))
			continue;
		    Value s =
		        atEntry(b, ulm_lam::dimValue(b, loc, v, d), entry);
		    if (!s)
			return call.emitError("ulm-lam-workspace: the size of "
			                      "a matrix argument is not "
			                      "available at function entry"),
			       failure();
		    sizes.push_back(s);
		}
		if (!c0)
		    c0 = arith::ConstantIndexOp::create(b, loc, 0);
		ops.push_back(
		    memref::ViewOp::create(b, loc, t, work, c0, sizes));
		continue;
	    }
	    if (auto a = dyn_cast<BlockArgument>(v);
	        a && a.getOwner() == &entry) {
		ops.push_back(v);
		continue;
	    }
	    if (Operation *def = v.getDefiningOp();
	        def && isa<arith::ConstantOp>(def)) {
		ops.push_back(b.clone(*def)->getResult(0));
		continue;
	    }
	    // not available at entry (e.g. a loop index): 0.  The requirement
	    // of a function with a workspace depends only on its (static)
	    // matrix sizes and on scalars available at entry.
	    Type t = v.getType();
	    if (isa<IndexType>(t))
		ops.push_back(arith::ConstantIndexOp::create(b, loc, 0));
	    else if (isa<IntegerType, FloatType>(t))
		ops.push_back(
		    arith::ConstantOp::create(b, loc, t, b.getZeroAttr(t)));
	    else
		return call.emitError("ulm-lam-workspace: cannot query the "
		                      "workspace of this call: unsupported "
		                      "argument type"),
		       failure();
	}
	return ops;
    }

    LogicalResult
    rewrite(func::FuncOp f, SymbolTable &symbols,
            const llvm::SetVector<func::FuncOp> &ws)
    {
	MLIRContext *ctx = f.getContext();
	Location loc = f.getLoc();
	unsigned n = f.getNumArguments();
	Value work = f.getArgument(n - 2), info = f.getArgument(n - 1);
	Block &entry = f.getBody().front();
	auto ret = cast<func::ReturnOp>(entry.getTerminator());

	// own temporaries
	SmallVector<memref::AllocOp> allocs;
	f.walk([&](memref::AllocOp a) { allocs.push_back(a); });
	for (memref::AllocOp a : allocs)
	    if (!a.getType().getLayout().isIdentity())
		return a.emitError(
		    "ulm-lam-workspace: only identity layout temporaries");
	// calls of functions with a workspace
	SmallVector<func::CallOp> calls;
	f.walk([&](func::CallOp c) {
	    if (auto g = symbols.lookup<func::FuncOp>(c.getCallee()))
		if (ws.contains(g))
		    calls.push_back(c);
	});

	SmallVector<Operation *> body;
	for (Operation &op : entry.without_terminator())
	    body.push_back(&op);

	OpBuilder b(ret);
	auto i64 = [&](int64_t v) {
	    return arith::ConstantIntOp::create(b, loc, v, 64).getResult();
	};
	auto toIndex = [&](Value v) {
	    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v)
	        .getResult();
	};
	Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
	Value given = memref::LoadOp::create(b, loc, info, ValueRange{c0});

	// layout of the own temporaries: [t1 | t2 | ...], each part rounded up
	// to the alignment; sizes from the dimensions (parameters at entry)
	// (in index arithmetic: size_t in C++; static parts fold)
	auto idx = [&](int64_t v) {
	    return arith::ConstantIndexOp::create(b, loc, v).getResult();
	};
	// A choice between plans (scf.if): both branches start at the same
	// offset; the size is the one of the branch that runs if its
	// condition can be computed at entry (the cost comparison of the
	// plans, from the sizes), otherwise the larger one.
	llvm::DenseMap<Operation *, Value> offsetOf;
	llvm::DenseMap<Value, Value> hoisted;
	std::function<Value(Value)> hoist = [&](Value v) -> Value {
	    if (auto a = dyn_cast<BlockArgument>(v))
		return a.getOwner() == &entry ? v : Value();
	    if (Value h = hoisted.lookup(v))
		return h;
	    Operation *def = v.getDefiningOp();
	    if (!def || def->getNumRegions() ||
	        !isa<arith::ArithDialect>(def->getDialect()) || !isPure(def))
		return {};
	    IRMapping map;
	    for (Value o : def->getOperands()) {
		Value h = hoist(o);
		if (!h)
		    return {};
		map.map(o, h);
	    }
	    Operation *c = b.clone(*def, map);
	    for (auto [x, y] : llvm::zip(def->getResults(), c->getResults()))
		hoisted[x] = y;
	    return hoisted.lookup(v);
	};
	auto bytesOf = [&](memref::AllocOp a) -> FailureOr<Value> {
	    MemRefType t = a.getType();
	    int64_t fixed = t.getElementTypeBitWidth() / 8;
	    for (int64_t d : t.getShape())
		if (!ShapedType::isDynamic(d))
		    fixed *= d;
	    Value bytes = idx(fixed);
	    for (Value s : a.getDynamicSizes()) {
		Value e = atEntry(b, s, entry);
		if (!e)
		    return a.emitError("ulm-lam-workspace: the size of a "
		                       "temporary is not available at function "
		                       "entry"),
		           failure();
		bytes = arith::MulIOp::create(b, loc, bytes, e);
	    }
	    return arith::AndIOp::create(
	               b, loc,
	               arith::AddIOp::create(b, loc, bytes, idx(kAlign - 1)),
	               idx(-kAlign))
	        .getResult();
	};
	std::function<FailureOr<Value>(Block &, Value)> layout =
	    [&](Block &blk, Value cur) -> FailureOr<Value> {
	    for (Operation &op : blk) {
		if (auto a = dyn_cast<memref::AllocOp>(op)) {
		    FailureOr<Value> n = bytesOf(a);
		    if (failed(n))
			return failure();
		    offsetOf[a] = cur;
		    cur = arith::AddIOp::create(b, loc, cur, *n);
		    continue;
		}
		if (!op.getNumRegions() ||
		    !op.walk([](memref::AllocOp) {
		           return WalkResult::interrupt();
		       }).wasInterrupted())
		    continue;
		if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
		    FailureOr<Value> t = layout(*ifOp.thenBlock(), cur);
		    FailureOr<Value> e = ifOp.elseBlock()
		                             ? layout(*ifOp.elseBlock(), cur)
		                             : FailureOr<Value>(cur);
		    if (failed(t) || failed(e))
			return failure();
		    Value c = hoist(ifOp.getCondition());
		    if (!c) // the larger one
			c = arith::CmpIOp::create(
			    b, loc, arith::CmpIPredicate::uge, *t, *e);
		    cur = arith::SelectOp::create(b, loc, c, *t, *e);
		    continue;
		}
		for (Region &r : op.getRegions())
		    for (Block &bb : r) {
			FailureOr<Value> n = layout(bb, cur);
			if (failed(n))
			    return failure();
			cur = *n;
		    }
	    }
	    return cur;
	};
	FailureOr<Value> ownSize = layout(entry, idx(0));
	if (failed(ownSize))
	    return failure();
	Value own = *ownSize;

	// query the callees (the accumulation of LAPACK workspace queries)
	SmallVector<Value> sizes, parts; // per call: queried size, its offset
	Value required =
	    arith::IndexCastOp::create(b, loc, b.getI64Type(), own);
	for (func::CallOp call : calls) {
	    FailureOr<SmallVector<Value>> ops =
	        queryOperands(b, call, entry, work);
	    if (failed(ops))
		return failure();
	    ops->push_back(work);
	    ops->push_back(info);
	    memref::StoreOp::create(b, loc, i64(-1), info, ValueRange{c0});
	    auto qc = func::CallOp::create(b, loc, call.getCallee(),
	                                   TypeRange{}, *ops);
	    qc->setAttr("ulm_lam.query", b.getUnitAttr()); // for ulm-lam-info
	    Value q = memref::LoadOp::create(b, loc, info, ValueRange{c0});
	    Value rounded = arith::AndIOp::create(
	        b, loc, arith::AddIOp::create(b, loc, q, i64(kAlign - 1)),
	        i64(-kAlign));
	    parts.push_back(required);
	    sizes.push_back(q);
	    required = arith::AddIOp::create(b, loc, required, rounded);
	}
	if (!calls.empty())
	    memref::StoreOp::create(b, loc, given, info, ValueRange{c0});

	Value isQuery = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
	                                      given, i64(-1));
	auto query = scf::IfOp::create(b, loc, isQuery, /*withElse=*/true);
	b.setInsertionPointToStart(query.thenBlock());
	memref::StoreOp::create(b, loc, required, info, ValueRange{c0});

	b.setInsertionPointToStart(query.elseBlock());
	Value positive = arith::CmpIOp::create(
	    b, loc, arith::CmpIPredicate::sgt, given, i64(0));
	Value tooSmall = arith::CmpIOp::create(
	    b, loc, arith::CmpIPredicate::slt, given, required);
	Value bad = arith::AndIOp::create(b, loc, positive, tooSmall);
	auto check = scf::IfOp::create(b, loc, bad, /*withElse=*/true);
	b.setInsertionPointToStart(check.thenBlock());
	func::CallOp::create(b, loc, kErrorFn, TypeRange{},
	                     ValueRange{required, given});

	// compute: the block is `work`, or allocated here (*info == 0)
	b.setInsertionPointToStart(check.elseBlock());
	Value self = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
	                                   given, i64(0));
	auto pick = scf::IfOp::create(b, loc, TypeRange{bufTy}, self, true);
	{
	    OpBuilder::InsertionGuard g(b);
	    b.setInsertionPointToStart(pick.thenBlock());
	    auto a = memref::AllocOp::create(b, loc, bufTy,
	                                     ValueRange{toIndex(required)});
	    a.setAlignment(kAlign);
	    scf::YieldOp::create(b, loc, a.getResult());
	    b.setInsertionPointToStart(pick.elseBlock());
	    scf::YieldOp::create(b, loc, work);
	}
	Value base = pick.getResult(0);
	Operation *anchor = b.getInsertionBlock()->getTerminator();
	for (Operation *op : body)
	    op->moveBefore(anchor);
	b.setInsertionPoint(anchor);
	if (!calls.empty()) // LAPACK leaves lwork unchanged
	    memref::StoreOp::create(b, loc, given, info, ValueRange{c0});
	auto release = scf::IfOp::create(b, loc, self, /*withElse=*/false);
	b.setInsertionPointToStart(release.thenBlock());
	memref::DeallocOp::create(b, loc, base);

	// own temporaries become views of the block
	for (memref::AllocOp a : allocs) {
	    Value off = offsetOf.lookup(a);
	    OpBuilder vb(a);
	    Value v = memref::ViewOp::create(vb, a.getLoc(), a.getType(), base,
	                                     off, a.getDynamicSizes());
	    for (Operation *u : llvm::to_vector(a->getUsers()))
		if (isa<memref::DeallocOp>(u))
		    u->erase();
	    a.getResult().replaceAllUsesWith(v);
	    a.erase();
	}
	// every call gets its part of the block and its size
	for (auto [call, size, part] : llvm::zip(calls, sizes, parts)) {
	    OpBuilder cb(call);
	    Location cl = call.getLoc();
	    Value z = arith::ConstantIndexOp::create(cb, cl, 0);
	    Value off =
	        arith::IndexCastOp::create(cb, cl, cb.getIndexType(), part);
	    Value len =
	        arith::IndexCastOp::create(cb, cl, cb.getIndexType(), size);
	    Value slot = memref::ViewOp::create(cb, cl, bufTy, base, off,
	                                        ValueRange{len});
	    memref::StoreOp::create(cb, cl, size, info, ValueRange{z});
	    SmallVector<Value> ops(call.getOperands());
	    ops.push_back(slot);
	    ops.push_back(info);
	    func::CallOp::create(cb, cl, call.getCallee(), TypeRange{}, ops);
	    call.erase();
	}
	(void)ctx;
	return success();
    }
};

} // namespace

namespace ulm_lam {
void
registerWorkspacePass()
{
    PassRegistration<UlmLamWorkspace>();
}
} // namespace ulm_lam
