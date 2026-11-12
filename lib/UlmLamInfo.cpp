//===- UlmLamInfo.cpp - LAPACK-style info results
//---------------------------===//
//
// ulm-lam-info: every defined function returns an info value, as a LAPACK
// routine does:
//
//   info == 0   success
//   info == k   the k-th check of the function failed: a factorisation
//               (potrf: the matrix is not positive definite, so the spd
//               assumption did not hold numerically; getrf: the matrix is
//               singular), or a call of another ulm.lam function that
//               returned info != 0
//   info == -i  argument i is illegal: the workspace given in
//               *workspace_info (argument i) is too small
//
// The checks are numbered in the order of the function body; their
// descriptions are kept in the function attribute `ulm_lam.info` (an array
// of strings, entry k-1 for check k) for the interface comment.
//
// After a check, the rest of its block runs only if the check passed:
//
//   %info = ulm_lamk.potrf ... outs(%A)
//   <rest>
//
// becomes
//
//   %info = ulm_lamk.potrf ... outs(%A)
//   %code = arith.select (%info != 0), k, 0
//   %s = scf.if (%code == 0) -> i64 { <rest>; yield <status of rest> }
//                          else { yield %code }
//
// A loop carries the status: an iteration runs only if no earlier one
// failed (the remaining iterations do nothing), and the loop's status is
// the final one.  A failing function returns at once, without finishing
// its work; like LAPACK, its outputs are then unspecified.  The call of
// ulm_lam_workspace_error that ulm-lam-workspace generates becomes the
// negative info; the function is no longer needed.
//
// Runs after ulm-lam-workspace (on memrefs).
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include <string>

using namespace mlir;
namespace lk = mlir::ulm_lamk;

namespace {

constexpr llvm::StringLiteral kErrorFn = "ulm_lam_workspace_error";
constexpr llvm::StringLiteral kInfoAttr = "ulm_lam.info";

struct UlmLamInfo : PassWrapper<UlmLamInfo, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamInfo)

    StringRef
    getArgument() const final
    {
	return "ulm-lam-info";
    }
    StringRef
    getDescription() const final
    {
	return "Every function returns a LAPACK-style info: 0, k > 0 (check "
	       "k failed: a factorisation or a call), or -i (argument i "
	       "illegal: workspace too small)";
    }
    void
    getDependentDialects(DialectRegistry &registry) const override
    {
	registry
	    .insert<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    }

    llvm::StringSet<> defined;
    /// check op -> its number k (code), per function
    llvm::DenseMap<Operation *, int64_t> codes;

    bool
    isDefinedCall(Operation *op) const
    {
	auto c = dyn_cast<func::CallOp>(op);
	return c && defined.contains(c.getCallee());
    }
    static bool
    isErrorCall(Operation *op)
    {
	auto c = dyn_cast<func::CallOp>(op);
	return c && c.getCallee() == kErrorFn;
    }
    bool
    isCheck(Operation *op) const
    {
	return isa<lk::PotrfOp, lk::GetrfOp>(op) || isDefinedCall(op) ||
	       isErrorCall(op);
    }
    bool
    containsCheck(Operation *op) const
    {
	return op
	    ->walk([&](Operation *o) {
	        return o != op && isCheck(o) ? WalkResult::interrupt()
		                             : WalkResult::advance();
	    })
	    .wasInterrupted();
    }

    void
    runOnOperation() override
    {
	ModuleOp module = getOperation();
	MLIRContext *ctx = &getContext();
	Type i64 = IntegerType::get(ctx, 64);

	SmallVector<func::FuncOp> funcs;
	for (auto f : module.getOps<func::FuncOp>())
	    if (!f.isExternal()) {
		funcs.push_back(f);
		defined.insert(f.getSymName());
	    }

	// 1. number the checks and describe them (before calls change)
	for (func::FuncOp f : funcs) {
	    SmallVector<Attribute> text;
	    f.walk<WalkOrder::PreOrder>([&](Operation *op) {
		// (a workspace query, generated at entry, is not numbered)
		if (!isCheck(op) || isErrorCall(op) ||
		    op->hasAttr("ulm_lam.query"))
		    return;
		codes[op] = text.size() + 1;
		text.push_back(StringAttr::get(ctx, describe(f, op)));
	    });
	    f->setAttr(kInfoAttr, ArrayAttr::get(ctx, text));
	}

	// 2. signatures: one more result
	for (func::FuncOp f : funcs) {
	    SmallVector<Type> results(f.getResultTypes());
	    results.push_back(i64);
	    f.setFunctionType(
	        FunctionType::get(ctx, f.getArgumentTypes(), results));
	}
	// ... and their calls
	SmallVector<func::CallOp> calls;
	module.walk([&](func::CallOp c) {
	    if (defined.contains(c.getCallee()))
		calls.push_back(c);
	});
	for (func::CallOp c : calls) {
	    OpBuilder b(c);
	    SmallVector<Type> results(c.getResultTypes());
	    results.push_back(i64);
	    auto n = func::CallOp::create(b, c.getLoc(), c.getCallee(), results,
	                                  c.getOperands());
	    n->setDiscardableAttrs(c->getDiscardableAttrDictionary());
	    for (auto [o, r] : llvm::zip(c.getResults(), n.getResults()))
		o.replaceAllUsesWith(r);
	    if (codes.contains(c)) {
		codes[n] = codes[c];
		codes.erase(c);
	    } else {
		// a workspace query (generated at entry): not numbered; a
		// query does not fail
		codes[n] = 0;
	    }
	    c.erase();
	}

	// 3. bodies
	for (func::FuncOp f : funcs) {
	    Block &entry = f.getBody().front();
	    FailureOr<Value> s = guard(entry, f);
	    if (failed(s))
		return signalPassFailure();
	    auto ret = cast<func::ReturnOp>(entry.getTerminator());
	    SmallVector<Value> ops(ret.getOperands());
	    ops.push_back(*s);
	    OpBuilder b(ret);
	    func::ReturnOp::create(b, ret.getLoc(), ops);
	    ret.erase();
	}

	// 4. the workspace error function is no longer called
	if (auto e = module.lookupSymbol<func::FuncOp>(kErrorFn))
	    if (SymbolTable::symbolKnownUseEmpty(e, module))
		e.erase();
    }

    // ---- descriptions ----------------------------------------------------

    /// "v3" for argument 2 of f, "a copy of v3" for a buffer filled from
    /// it, otherwise "a computed matrix".
    static std::string
    nameOf(func::FuncOp f, Value m)
    {
	auto argName = [&](Value v) -> std::string {
	    if (auto a = dyn_cast<BlockArgument>(v);
	        a && a.getOwner() == &f.getBody().front())
		return "v" + std::to_string(a.getArgNumber() + 1);
	    return "";
	};
	if (std::string n = argName(m); !n.empty())
	    return n;
	// a kernel other than the factorisation that writes m (e.g. adddiag:
	// K + s^2 I is factorised, not K)
	bool changed = llvm::any_of(m.getUsers(), [&](Operation *u) {
	    auto d = dyn_cast<DestinationStyleOpInterface>(u);
	    return d && !isa<ulm_lamk::PotrfOp, ulm_lamk::GetrfOp>(u) &&
	           llvm::any_of(d.getDpsInits(),
	                        [&](Value v) { return v == m; });
	});
	for (Operation *u : m.getUsers())
	    if (auto c = dyn_cast<memref::CopyOp>(u); c && c.getTarget() == m)
		if (std::string n = argName(c.getSource()); !n.empty())
		    // a copy of it (the function works on a copy)
		    return changed ? "a matrix computed from " + n : n;
	return "a computed matrix";
    }

    std::string
    describe(func::FuncOp f, Operation *op) const
    {
	if (auto p = dyn_cast<lk::PotrfOp>(op))
	    return "Cholesky factorisation of " + nameOf(f, p.getA()) +
	           ": not positive definite";
	if (auto p = dyn_cast<lk::GetrfOp>(op))
	    return "LU factorisation of " + nameOf(f, p.getA()) + ": singular";
	auto c = cast<func::CallOp>(op);
	return "call of " + c.getCallee().str() + " returned info != 0";
    }

    // ---- guards ------------------------------------------------------------

    /// The failure code of check `op`, computed right after it.
    Value
    codeOf(OpBuilder &b, Operation *op, func::FuncOp f)
    {
	Location loc = op->getLoc();
	Type i64 = b.getI64Type();
	auto cst = [&](int64_t v) {
	    return arith::ConstantIntOp::create(b, loc, i64, v).getResult();
	};
	if (isErrorCall(op)) // -i: argument i (*workspace_info) is illegal
	    return cst(-static_cast<int64_t>(f.getNumArguments()));
	Value info = op->getResults().back();
	int64_t k = codes.lookup(op);
	if (k == 0) // a workspace query: its info is passed on
	    return info;
	Value failed = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
	                                     info, cst(0));
	return arith::SelectOp::create(b, loc, failed, cst(k), cst(0));
    }

    /// Give region op `op` (scf.if, scf.for) containing checks a status
    /// result; returns it.
    FailureOr<Value>
    statusOf(Operation *op, func::FuncOp f)
    {
	Type i64 = IntegerType::get(op->getContext(), 64);
	if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
	    OpBuilder b(ifOp);
	    SmallVector<Type> types(ifOp.getResultTypes());
	    types.push_back(i64);
	    auto n = scf::IfOp::create(b, ifOp.getLoc(), types,
	                               ifOp.getCondition(), /*withElse=*/true);
	    n.getThenRegion().takeBody(ifOp.getThenRegion());
	    if (!ifOp.getElseRegion().empty()) {
		n.getElseRegion().takeBody(ifOp.getElseRegion());
	    } else {
		OpBuilder eb = OpBuilder::atBlockEnd(n.elseBlock());
		scf::YieldOp::create(eb, ifOp.getLoc());
	    }
	    for (Block *blk : {n.thenBlock(), n.elseBlock()}) {
		FailureOr<Value> s = guard(*blk, f);
		if (failed(s))
		    return failure();
		Operation *y = blk->getTerminator();
		y->insertOperands(y->getNumOperands(), *s);
	    }
	    for (auto [o, r] : llvm::zip(ifOp.getResults(), n.getResults()))
		o.replaceAllUsesWith(r);
	    ifOp.erase();
	    return n.getResults().back();
	}
	if (auto loop = dyn_cast<scf::ForOp>(op)) {
	    IRRewriter rw(op->getContext());
	    rw.setInsertionPoint(loop);
	    Value zero =
	        arith::ConstantIntOp::create(rw, loop.getLoc(), i64, 0);
	    // the status enters as an iteration argument; placeholder yield
	    // (replaced below)
	    FailureOr<LoopLikeOpInterface> r = loop.replaceWithAdditionalYields(
	        rw, ValueRange{zero}, /*replaceInitOperandUsesInLoop=*/false,
	        [&](OpBuilder &, Location, ArrayRef<BlockArgument> args) {
		    return SmallVector<Value>{args.back()};
	        });
	    if (failed(r))
		return loop.emitError("ulm-lam-info: cannot extend the loop");
	    auto n = cast<scf::ForOp>(r->getOperation());
	    Block *body = n.getBody();
	    Value in = body->getArguments().back();
	    // an iteration runs only if no earlier one failed
	    Operation *y = body->getTerminator();
	    y->eraseOperand(y->getNumOperands() - 1);
	    FailureOr<Value> s = guardFrom(*body, nullptr, in, f);
	    if (failed(s))
		return failure();
	    y = body->getTerminator();
	    y->insertOperands(y->getNumOperands(), *s);
	    return n.getResults().back();
	}
	return op->emitError("ulm-lam-info: a check inside an unsupported "
	                     "region op"),
	       failure();
    }

    /// Transform `blk` so that every check guards the rest of the block;
    /// returns the status of the block (an i64 defined in it).
    FailureOr<Value>
    guard(Block &blk, func::FuncOp f)
    {
	for (Operation *op = &blk.front(); op != blk.getTerminator();
	     op = op->getNextNode()) {
	    if (isCheck(op)) {
		OpBuilder b(op->getContext());
		b.setInsertionPointAfter(op);
		Value code = codeOf(b, op, f);
		Operation *after = op;
		if (Operation *d = code.getDefiningOp(); d && d != op)
		    after = d;
		if (isErrorCall(op))
		    op->erase();
		return guardFrom(blk, after, code, f);
	    }
	    if (op->getNumRegions() && containsCheck(op)) {
		FailureOr<Value> s = statusOf(op, f);
		if (failed(s))
		    return failure();
		return guardFrom(blk, s->getDefiningOp(), *s, f);
	    }
	}
	OpBuilder b(blk.getTerminator());
	return arith::ConstantIntOp::create(b, blk.getTerminator()->getLoc(),
	                                    b.getI64Type(), 0)
	    .getResult();
    }

    /// memref.dealloc, or an scf.if that only deallocates (the release of
    /// a workspace the function allocated itself).
    static bool
    isRelease(Operation *op)
    {
	if (isa<memref::DeallocOp>(op))
	    return true;
	auto i = dyn_cast<scf::IfOp>(op);
	if (!i || i.getNumResults() != 0)
	    return false;
	return llvm::all_of(i->getRegions(), [](Region &r) {
	    return llvm::all_of(r.getOps(), [](Operation &o) {
		return isa<memref::DeallocOp, scf::YieldOp>(o);
	    });
	});
    }

    /// A value of type `t` defined before `after` in `blk` or in an
    /// enclosing block (before the op that contains it), or null.
    static Value
    earlierOfType(Block &blk, Operation *after, Type t)
    {
	Block *b = &blk;
	Operation *pos = after;
	while (b) {
	    for (Operation *op = pos; op; op = op->getPrevNode())
		for (Value r : op->getResults())
		    if (r.getType() == t)
			return r;
	    for (BlockArgument a : b->getArguments())
		if (a.getType() == t)
		    return a;
	    Operation *parent = b->getParentOp();
	    if (!parent || isa<func::FuncOp>(parent))
		break;
	    pos = parent->getPrevNode();
	    b = parent->getBlock();
	}
	return {};
    }

    /// The ops of `blk` after `after` (all ops if null) run only if `code`
    /// is 0: they move into an scf.if whose results are the status and the
    /// values the terminator needs from them.
    FailureOr<Value>
    guardFrom(Block &blk, Operation *after, Value code, func::FuncOp f)
    {
	Operation *term = blk.getTerminator();
	SmallVector<Operation *> rest;
	for (Operation *op = after ? after->getNextNode() : &blk.front();
	     op != term; op = op->getNextNode())
	    rest.push_back(op);
	// releasing memory at the end runs also after a failure, unless it
	// releases what the guarded ops allocate (then it is guarded too)
	SmallVector<Operation *> releases;
	while (!rest.empty() && isRelease(rest.back())) {
	    releases.push_back(rest.back());
	    rest.pop_back();
	}
	if (rest.empty())
	    return code;
	{
	    llvm::SmallPtrSet<Operation *, 16> body(rest.begin(), rest.end());
	    auto inBody = [&](Value v) {
		for (Operation *d = v.getDefiningOp(); d; d = d->getParentOp())
		    if (body.contains(d))
			return true;
		return false;
	    };
	    SmallVector<Operation *> keep;
	    for (Operation *r : llvm::reverse(releases)) {
		bool uses = false;
		r->walk([&](Operation *o) {
		    for (Value v : o->getOperands())
			uses |= inBody(v);
		});
		if (uses)
		    rest.push_back(r);
		else
		    keep.push_back(r);
	    }
	    // (the kept releases stay after the guarded ops)
	    for (Operation *r : keep)
		if (!rest.empty() && rest.back()->isBeforeInBlock(r))
		    continue;
	}
	Operation *anchor = rest.back()->getNextNode();
	llvm::SmallPtrSet<Operation *, 16> moved(rest.begin(), rest.end());
	// values of the moved ops that the terminator uses, and what it gets
	// instead if the check failed
	SmallVector<unsigned> idx;
	SmallVector<Value> inner, otherwise;
	for (OpOperand &o : term->getOpOperands()) {
	    Operation *def = o.get().getDefiningOp();
	    if (!def || !moved.contains(def))
		continue;
	    auto loop = dyn_cast<scf::ForOp>(blk.getParentOp());
	    // after a failed check: the value from before the iteration (a
	    // loop), or any value of the type defined before the check (a
	    // branch: its content is not used, the function returns the code)
	    Value alt;
	    if (isa<scf::YieldOp>(term) && loop)
		alt = loop.getRegionIterArgs()[o.getOperandNumber()];
	    else if (isa<scf::YieldOp>(term))
		alt = earlierOfType(blk, after, o.get().getType());
	    if (!alt)
		return term->emitError("ulm-lam-info: cannot guard a block "
		                       "whose result is computed after a "
		                       "check"),
		       failure();
	    idx.push_back(o.getOperandNumber());
	    inner.push_back(o.get());
	    otherwise.push_back(alt);
	}
	OpBuilder b(anchor);
	Location loc = term->getLoc();
	Value ok = arith::CmpIOp::create(
	    b, loc, arith::CmpIPredicate::eq, code,
	    arith::ConstantIntOp::create(b, loc, b.getI64Type(), 0));
	SmallVector<Type> types{b.getI64Type()};
	for (Value v : inner)
	    types.push_back(v.getType());
	auto ifOp = scf::IfOp::create(b, loc, types, ok, /*withElse=*/true);
	// then: the rest, and its status
	Block *then = ifOp.thenBlock();
	for (Operation *op : rest)
	    op->moveBefore(then, then->end());
	OpBuilder tb = OpBuilder::atBlockEnd(then);
	auto ty = scf::YieldOp::create(tb, loc, ValueRange{});
	FailureOr<Value> s = guard(*then, f);
	if (failed(s))
	    return failure();
	SmallVector<Value> thenVals{*s};
	llvm::append_range(thenVals, inner);
	ty->setOperands(thenVals);
	// else: the code, and the values from before the iteration
	OpBuilder eb = OpBuilder::atBlockEnd(ifOp.elseBlock());
	SmallVector<Value> elseVals{code};
	llvm::append_range(elseVals, otherwise);
	scf::YieldOp::create(eb, loc, elseVals);
	for (auto [k, i] : llvm::enumerate(idx))
	    term->setOperand(i, ifOp.getResult(k + 1));
	return ifOp.getResult(0);
    }
};

} // namespace

namespace ulm_lam {
void
registerInfoPass()
{
    PassRegistration<UlmLamInfo>();
}
} // namespace ulm_lam
