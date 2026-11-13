//===- UlmLamPasses.cpp - normal form, inference, kernel selection
//---------===//
//
//   ulm-lam-normalize         greedy ulm_lam canonicalisation patterns
//   ulm-lam-infer-properties  strengthen result types to the inferred ones
//   ulm-lam-select-kernels    ulm_lam -> ulm_lamk (strategy=gmc)
//   ulm-lam-strip-properties  drop #ulm_lam.props encodings and ulm_lam.assume
//
// ulm-lam-select-kernels, strategy=gmc:
//
//   Inverses.  Every inverted matrix M is factorised once, right after its
//   definition (so the factorisation is shared by all products and stays
//   outside loops in which M is invariant), chosen by its properties:
//     lower/upper triangular  M itself              M^-1 -> solve with M
//     spd                     potrf: M = L L^T      M^-1 -> L^-T L^-1
//     otherwise               getrf: P M = L U      M^-1 -> LU solve
//   An inverse is never formed explicitly.
//
//   Products.  A product of leaves (matrices, possibly transposed, and
//   solves) is parenthesised by the generalised matrix chain dynamic
//   programme over FLOPs.  Combinations: gemm (gemv for a vector), solve
//   from the left (trsm / trsv / getrs on the right operand, in place),
//   triangular solve from the right (trsm side right), and Z^T Z for an
//   interval that is its own transpose (Linnea's common subexpression up to
//   transposition: X^T L^-T L^-1 X = (L^-1 X)^T (L^-1 X)).  Transposition is
//   always a kernel flag, never a copy.  An interval that an earlier
//   product has computed (W = L^-1 X), directly or transposed (W^T =
//   X^T L^-T), costs nothing if its value dominates the product.
//
//   Sums.  Y_1 + a_2 Y_2 + ... + b_1 P_1 + ... accumulates into the first
//   plain term: gescal for its scale, geaxpy for the other plain terms, and
//   the last kernel of every product writes into the accumulator (gemm with
//   beta = 1; solves via geaxpy).  X = X + A B C therefore updates X in
//   place after bufferization.
//
//   Constant factors of a product (lam.scale) move into alpha.
//
//   Diagnosed instead of miscompiled: explicit inverses (an inverse that is
//   not a factor of a product), right solves with an LU factorisation,
//   transposed accumulators, transposes that are not kernel operands,
//   dynamic shapes.
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/SaveAndRestore.h"

#include <functional>
#include <limits>
#include <map>
#include <optional>

using namespace mlir;
using namespace mlir::ulm_lam;
namespace lk = mlir::ulm_lamk;

namespace {

bool
isUlmLamOp(Operation *op)
{
    return isa_and_nonnull<UlmLamDialect>(op->getDialect());
}

/// x' = assume(x) that only drops properties (as folds leave behind to
/// keep a result type, e.g. transpose(transpose(P)) -> assume(P)): its
/// uses in ulm_lam operations get x itself, which has more properties, and
/// which other rules then recognise (P and P^T as each other's transpose,
/// the full rank of P).  Other uses (return, calls) keep x'.
/// assume(assume(x)) back to the type of x is x (a value whose properties
/// were dropped and asserted again, as a call from Julia leaves behind for a
/// symmetric matrix: transpose(S) -> S without properties -> the callee's
/// type).
struct FoldAssumeRoundTrip : OpRewritePattern<AssumeOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(AssumeOp op, PatternRewriter &rewriter) const override
    {
	auto inner = op.getOperand().getDefiningOp<AssumeOp>();
	if (!inner || op.getChecked() || inner.getChecked() ||
	    inner.getOperand().getType() != op.getType())
	    return failure();
	rewriter.replaceOp(op, inner.getOperand());
	return success();
    }
};

struct ForwardDroppingAssume : OpRewritePattern<AssumeOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(AssumeOp op, PatternRewriter &rewriter) const override
    {
	if (op.getChecked() ||
	    !bitEnumContainsAll(propertiesOf(op.getOperand().getType()),
	                        propertiesOf(op.getType())))
	    return failure();
	bool changed = false;
	for (OpOperand &u : llvm::make_early_inc_range(op->getUses())) {
	    Operation *user = u.getOwner();
	    if (!isa<MulOp, AddOp, TransposeOp, InverseOp, ScaleOp>(user))
		continue;
	    rewriter.modifyOpInPlace(user, [&] { u.set(op.getOperand()); });
	    changed = true;
	}
	return success(changed);
    }
};

//===----------------------------------------------------------------------===//
// ulm-lam-normalize
//===----------------------------------------------------------------------===//

/// A loop whose state starts as transpose(P) carries P instead: the body
/// reads transpose(state) and yields transpose(new state), and the
/// transposes are pushed to the factors (PushTranspose), where they cost
/// nothing (a flag of gemm).  The Julia frontend hands every matrix over
/// transposed (a column-major array is a transposed row-major one); a
/// loop carrying it so would copy it transposed on entry and exit.
struct FlipTransposedLoopState : OpRewritePattern<scf::ForOp>
{
    using OpRewritePattern::OpRewritePattern;

    static RankedTensorType
    swapped(Type t)
    {
	auto r = cast<RankedTensorType>(t);
	return RankedTensorType::get({r.getDimSize(1), r.getDimSize(0)},
	                             r.getElementType());
    }

    /// x as type t (the same shape): an assume where the properties differ
    static Value
    as(PatternRewriter &rw, Location loc, Value x, Type t)
    {
	return x.getType() == t
	           ? x
		   : AssumeOp::create(rw, loc, t, x, false).getResult();
    }

    static Value
    transposed(PatternRewriter &rw, Location loc, Value x, Type t)
    {
	Value r = TransposeOp::create(rw, loc, swapped(x.getType()), x);
	return as(rw, loc, r, t);
    }

    LogicalResult
    matchAndRewrite(scf::ForOp loop, PatternRewriter &rw) const override
    {
	SmallVector<bool> flip;
	bool any = false;
	for (auto [init, arg] :
	     llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs())) {
	    auto t = dyn_cast<RankedTensorType>(init.getType());
	    // (not a state the body declares symmetric: its orientation does
	    // not matter, P = assume(P, :symmetric) in a Kalman filter)
	    bool sym = llvm::any_of(arg.getUsers(), [](Operation *u) {
		auto as = dyn_cast<AssumeOp>(u);
		if (!as)
		    return false;
		auto at = cast<RankedTensorType>(as.getType());
		return bitEnumContainsAll(
		    closure(propertiesOf(at), at.getShape()),
		    Properties::symmetric);
	    });
	    bool f = t && t.getRank() == 2 && !sym &&
	             init.getDefiningOp<TransposeOp>();
	    flip.push_back(f);
	    any |= f;
	}
	if (!any)
	    return failure();
	Location loc = loop.getLoc();
	SmallVector<Value> inits;
	for (auto [k, init] : llvm::enumerate(loop.getInitArgs())) {
	    if (!flip[k]) {
		inits.push_back(init);
		continue;
	    }
	    Value p = init.getDefiningOp<TransposeOp>().getOperand();
	    inits.push_back(as(rw, loc, p, swapped(init.getType())));
	}
	auto nl = scf::ForOp::create(
	    rw, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
	    inits, [&](OpBuilder &, Location, Value, ValueRange) {});
	Block *ob = loop.getBody(), *nb = nl.getBody();
	IRMapping map;
	map.map(loop.getInductionVar(), nl.getInductionVar());
	rw.setInsertionPointToStart(nb);
	for (auto [k, oa] : llvm::enumerate(loop.getRegionIterArgs())) {
	    Value na = nl.getRegionIterArgs()[k];
	    map.map(oa, flip[k] ? transposed(rw, loc, na, oa.getType()) : na);
	}
	for (Operation &op : ob->without_terminator())
	    rw.clone(op, map);
	auto yield = cast<scf::YieldOp>(ob->getTerminator());
	SmallVector<Value> ys;
	for (auto [k, y] : llvm::enumerate(yield.getOperands())) {
	    Value v = map.lookupOrDefault(y);
	    ys.push_back(flip[k]
	                     ? transposed(rw, loc, v,
			                  nl.getRegionIterArgs()[k].getType())
			     : v);
	}
	// (the builder's empty body has a yield already)
	if (!nb->empty() && isa<scf::YieldOp>(nb->back()))
	    rw.eraseOp(&nb->back());
	scf::YieldOp::create(rw, loc, ys);
	rw.setInsertionPointAfter(nl);
	SmallVector<Value> results;
	for (auto [k, r] : llvm::enumerate(nl.getResults()))
	    results.push_back(
	        flip[k] ? transposed(rw, loc, r, loop.getResult(k).getType())
		        : r);
	rw.replaceOp(loop, results);
	return success();
    }
};

struct UlmLamNormalize
    : PassWrapper<UlmLamNormalize, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamNormalize)
    StringRef
    getArgument() const final
    {
	return "ulm-lam-normalize";
    }
    StringRef
    getDescription() const final
    {
	return "Bring ulm_lam expressions into normal form";
    }
    void
    runOnOperation() override
    {
	MLIRContext *ctx = &getContext();
	RewritePatternSet patterns(ctx);
	patterns.add<ForwardDroppingAssume, FoldAssumeRoundTrip,
	             FlipTransposedLoopState>(ctx);
	MulOp::getCanonicalizationPatterns(patterns, ctx);
	AddOp::getCanonicalizationPatterns(patterns, ctx);
	ScaleOp::getCanonicalizationPatterns(patterns, ctx);
	TransposeOp::getCanonicalizationPatterns(patterns, ctx);
	InverseOp::getCanonicalizationPatterns(patterns, ctx);
	if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
	    signalPassFailure();
    }
};

//===----------------------------------------------------------------------===//
// ulm-lam-infer-properties
//===----------------------------------------------------------------------===//

struct UlmLamInferProperties
    : PassWrapper<UlmLamInferProperties, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamInferProperties)
    StringRef
    getArgument() const final
    {
	return "ulm-lam-infer-properties";
    }
    StringRef
    getDescription() const final
    {
	return "Refine #ulm_lam.props encodings to the strongest inferable";
    }
    void
    runOnOperation() override
    {
	// Program order: operands are refined before their users.
	getOperation().walk([&](InferTypeOpInterface op) {
	    if (!isUlmLamOp(op) || op->getNumResults() != 1)
		return;
	    SmallVector<Type> inferred;
	    if (failed(op.inferReturnTypes(
	            op->getContext(), op->getLoc(), op->getOperands(),
	            op->getDiscardableAttrDictionary(),
	            op->getPropertiesStorage(), op->getRegions(), inferred)))
		return;
	    Value r = op->getResult(0);
	    Type old = r.getType();
	    if (inferred[0] == old)
		return;
	    r.setType(inferred[0]);
	    // Non-ulm_lam users keep seeing the old type.
	    OpBuilder b(op->getContext());
	    b.setInsertionPointAfter(op);
	    Value cast;
	    for (OpOperand &use : llvm::make_early_inc_range(r.getUses())) {
		if (isUlmLamOp(use.getOwner()))
		    continue;
		if (!cast)
		    cast = AssumeOp::create(b, op->getLoc(), old, r, false);
		use.set(cast);
	    }
	});
    }
};

//===----------------------------------------------------------------------===//
// ulm-lam-select-kernels
//===----------------------------------------------------------------------===//

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The size assumed for a dynamic dimension when costs are compared at
/// compile time.
constexpr double kNominal = 1000;

/// A dimension for the cost model.
double
costSize(int64_t d)
{
    return ShapedType::isDynamic(d) ? kNominal : static_cast<double>(d);
}

/// The stored triangle of a matrix value (lower_stored / upper_stored).
int
storageOf(Value v)
{
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t)
	return 0;
    // a kernel result has the type of its destination, but the kernel
    // decides what it holds (sycomplete: the whole matrix)
    if (Operation *def = v.getDefiningOp();
        def && isa<lk::UlmLamkDialect>(def->getDialect()))
	return 0;
    Properties p = propertiesOf(t);
    if (bitEnumContainsAll(p, Properties::lower_stored))
	return 1;
    if (bitEnumContainsAll(p, Properties::upper_stored))
	return 2;
    return 0;
}

/// A factor of a product after inverses have been replaced by solves.
struct Leaf
{
    enum Kind
    {
	Mat,
	TriSolve,
	LUSolve
    } kind = Mat;
    Value v;   // Mat: the matrix; TriSolve: triangular factor; LUSolve: LU
    Value piv; // LUSolve: pivots
    bool trans = false; // Mat: op(v) = v^T ; solves: op(v)^-1 = v^-T
    bool lower = true;  // TriSolve: v is lower triangular
    bool sym = false;   // LUSolve: the factors are L D L^T (sytrf)
    /// Mat: a symmetric matrix of which only one triangle is stored
    /// (lower_stored / upper_stored); never transposed (S^T = S).
    enum Storage
    {
	Full,
	Lower,
	Upper
    } stored = Full;

    RankedTensorType
    type() const
    {
	return cast<RankedTensorType>(v.getType());
    }
    /// The size of the rows / columns of op(v): a constant, or the index
    /// value of a dynamic dimension.
    OpFoldResult
    rowsDim() const
    {
	return dimOf(v, trans ? 1 : 0);
    }
    OpFoldResult
    colsDim() const
    {
	return dimOf(v, trans ? 0 : 1);
    }
    int64_t
    rows() const
    {
	return type().getDimSize(trans ? 1 : 0);
    }
    int64_t
    cols() const
    {
	return type().getDimSize(trans ? 0 : 1);
    }
    bool
    isSolve() const
    {
	return kind != Mat;
    }
    /// The same leaf transposed (for Z^T Z detection).
    bool
    isTransposeOf(const Leaf &o) const
    {
	if (kind == o.kind && v == o.v && piv == o.piv && lower == o.lower &&
	    trans != o.trans)
	    return true;
	// a symmetric matrix is its own transpose (P in (H P)^T = P H^T)
	if (kind != Mat || o.kind != Mat || v != o.v || trans != o.trans)
	    return false;
	auto t = cast<RankedTensorType>(v.getType());
	return bitEnumContainsAll(closure(propertiesOf(t), t.getShape()),
	                          Properties::symmetric);
    }
};

/// How an interval of leaves is computed.
struct Choice
{
    enum Kind
    {
	None,
	Gemm,
	SolveLeft,
	SolveRight,
	Gram,
	Reuse
    } kind = None;
    int split = 0; // Gemm/SolveLeft/SolveRight: last leaf of the left part;
                   // Gram: first leaf of Z
    Value value;   // Reuse: the product computed by an earlier product
    bool trans = false; // Reuse: used transposed
};

/// The factorisation used for the inverses of one matrix.
struct Factor
{
    enum Kind
    {
	Triangular,
	Cholesky,
	LU
    } kind;
    Value f;   // triangular matrix, L, or LU
    Value piv; // LU
    bool lower = true;
    bool sym = false; // LU: L D L^T of a symmetric matrix (sytrf)
};

class Selector
{
  public:
    explicit Selector(func::FuncOp f, bool lapack = false, bool expand = true)
        : func(f), b(f.getContext()), dom(f), lapack(lapack), expand(expand)
    {
    }
    /// expandFactors: computed factors of a product may be expanded
    bool expand = true;
    /// distribute: a product with a sum as a factor may be distributed
    bool distributeSums = true;
    /// The backend has LAPACK (CBLAS/LAPACK): L D L^T (sytrf) for symmetric
    /// matrices that are not known to be spd, instead of LU; least squares
    /// by QR (gels) instead of the normal equations.
    bool lapack = false;

    LogicalResult
    run()
    {
	if (!lapack)
	    normalEquations();
	if (distributeSums) {
	    woodbury();
	    distribute();
	}
	formIdentities();
	formInverses();
	SmallVector<Operation *> ops;
	func.walk([&](Operation *op) {
	    if (isa<MulOp, AddOp, LstsqOp>(op) || dynamicScale(op))
		ops.push_back(op);
	});
	for (Operation *op : ops)
	    if (auto m = dyn_cast<MulOp>(op); m && invertFactorwise(m))
		expanded.insert(m);
	for (Operation *op : ops) {
	    // a product used only by a sum is emitted by that sum; a product
	    // inverted factor by factor is never formed
	    if (consumed.contains(op) || feedsSum(op) ||
	        (expanded.contains(op) &&
	         llvm::all_of(op->getUsers(),
	                      [](Operation *u) { return isa<InverseOp>(u); })))
		continue;
	    b.setInsertionPoint(op);
	    if (auto ls = dyn_cast<LstsqOp>(op)) {
		replace(op, selectLstsq(ls));
		continue;
	    }
	    if (auto sc = dyn_cast<ScaleOp>(op)) {
		// (gescal scales both triangles: one stored triangle stays so)
		replace(op,
		        scaleBy(sc.getAlpha(), sc.getOperand(), op->getLoc()));
		continue;
	    }
	    FailureOr<Value> r = isa<AddOp>(op)
	                             ? selectAdd(cast<AddOp>(op))
	                             : selectProduct(cast<MulOp>(op));
	    if (failed(r))
		return failure();
	    if (auto m = dyn_cast<MulOp>(op))
		origin[*r] = m;
	    replace(op, *r);
	}
	// identities formed for inverses that became explicit (adddiag 1 on
	// gescal 0 on tensor.empty): erased once unused
	for (Value e : unusedIdentities) {
	    // (the replaced product A^-1 E itself)
	    for (Operation *u : llvm::make_early_inc_range(e.getUsers()))
		if (isa<MulOp>(u) && u->use_empty())
		    eraseDead(u);
	    while (e && e.use_empty()) {
		Operation *d = e.getDefiningOp();
		if (!d || !isa<lk::AdddiagOp, lk::GescalOp, tensor::EmptyOp>(d))
		    break;
		Value next;
		if (auto dps = dyn_cast<DestinationStyleOpInterface>(d))
		    next = dps.getDpsInits()[0];
		d->erase();
		e = next;
	    }
	}
	// A loop-carried value that is not computed in the buffer of its
	// iteration argument (x = F x cannot be in place) is copied into it at
	// the end of the body: One-Shot Bufferize needs the yielded value
	// equivalent to the iteration argument, otherwise it would allocate in
	// every iteration.
	func.walk([&](scf::ForOp loop) {
	    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
	    for (auto [k, o] : llvm::enumerate(yield->getOpOperands())) {
		Value v = o.get(), arg = loop.getRegionIterArg(k);
		if (!isa<RankedTensorType>(v.getType()) || v == arg)
		    continue;
		b.setInsertionPoint(yield);
		Value r = into(v, arg, yield.getLoc());
		// (the yield has the type of the loop's state; a value stored
		// as one triangle is completed before the copy, see below)
		if (r.getType() != arg.getType())
		    r = AssumeOp::create(b, yield.getLoc(), arg.getType(), r,
		                         false);
		o.set(r);
	    }
	});
	// A transpose used outside the ulm_lam ops (a loop's initial value, a
	// result, an argument of a call) is materialised: a transposed copy
	// (gecopy).  Inside products and sums transposes are strides.
	SmallVector<TransposeOp> transposes;
	func.walk([&](TransposeOp t) { transposes.push_back(t); });
	for (TransposeOp t : transposes) {
	    SmallVector<OpOperand *> outside;
	    for (OpOperand &u : t->getUses())
		if (!isUlmLamOp(u.getOwner()))
		    outside.push_back(&u);
	    if (outside.empty())
		continue;
	    b.setInsertionPointAfter(t);
	    Location loc = t.getLoc();
	    Leaf l;
	    l.v = t.getOperand();
	    l.trans = true;
	    Value e = empty(l, l, elementType(l.v), loc);
	    Value g = lk::GecopyOp::create(b, loc, TypeRange{e.getType()},
	                                   lk::Trans::trans, l.v, e)
	                  ->getResult(0);
	    if (g.getType() != t.getType())
		g = AssumeOp::create(b, loc, t.getType(), g, false);
	    for (OpOperand *u : outside)
		u->set(g);
	}
	// Other uses (return, calls, loops) get the whole matrix: the storage
	// of a matrix ends with the ops that know it.
	SmallVector<OpOperand *> uses;
	// (Also an ulm_lam.assume to a type without the storage: it would
	// drop it.)
	func.walk([&](Operation *op) {
	    if (auto as = dyn_cast<AssumeOp>(op)) {
		if (storageOf(as.getOperand()) && !storageOf(as.getResult()))
		    uses.push_back(&as->getOpOperand(0));
		return;
	    }
	    if (isUlmLamOp(op) || isa<lk::UlmLamkDialect>(op->getDialect()))
		return;
	    for (OpOperand &o : op->getOpOperands())
		if (storageOf(o.get()))
		    uses.push_back(&o);
	});
	for (OpOperand *o : uses) {
	    // (a result declared with the same storage, the new value of an
	    // inout parameter stored as one triangle: kept so)
	    if (auto ret = dyn_cast<func::ReturnOp>(o->getOwner())) {
		auto want = cast<RankedTensorType>(
		    func.getFunctionType().getResult(o->getOperandNumber()));
		Properties wp = propertiesOf(want);
		int st = storageOf(o->get());
		if ((st == 1 &&
		     bitEnumContainsAll(wp, Properties::lower_stored)) ||
		    (st == 2 &&
		     bitEnumContainsAll(wp, Properties::upper_stored)))
		    continue;
	    }
	    Value v = full(o->get());
	    // a result has the type the function declares (completed, the
	    // storage is no longer part of it)
	    if (auto ret = dyn_cast<func::ReturnOp>(o->getOwner())) {
		Type want =
		    func.getFunctionType().getResult(o->getOperandNumber());
		if (v.getType() != want) {
		    b.setInsertionPoint(ret);
		    v = AssumeOp::create(b, ret.getLoc(), want, v, false);
		}
	    }
	    o->set(v);
	}
	return cleanup();
    }

  private:
    func::FuncOp func;
    OpBuilder b;
    llvm::SmallPtrSet<Operation *, 8> consumed;
    /// Products that are only used inverted and are cheaper to invert
    /// factor by factor: (A B)^-1 = B^-1 A^-1.
    llvm::SmallPtrSet<Operation *, 8> expanded;
    llvm::DenseMap<Value, Factor> factors;
    DominanceInfo dom;

    /// Products of leaf sequences computed so far (as plain products:
    /// alpha = 1, beta = 0), for reuse by later products.  Keyed by the
    /// leaves; a later product can also use one transposed.
    using Key = std::vector<intptr_t>;
    std::map<Key, Value> computed;
    /// Computed products whose every use is expanded (expandFactors): they
    /// are not formed, so no chain may reuse them.
    llvm::DenseSet<Value> unreusable;

    static Key
    keyOf(ArrayRef<Leaf> leaves, int i, int j, bool transposed)
    {
	Key k;
	for (int t = 0; t <= j - i; ++t) {
	    const Leaf &l = transposed ? leaves[j - t] : leaves[i + t];
	    k.push_back(l.kind);
	    k.push_back(reinterpret_cast<intptr_t>(l.v.getAsOpaquePointer()));
	    k.push_back(reinterpret_cast<intptr_t>(l.piv.getAsOpaquePointer()));
	    // a symmetric matrix is its own transpose: P H^T is the transpose
	    // of a computed H P
	    auto ty = cast<RankedTensorType>(l.v.getType());
	    bool sym =
	        l.kind == Leaf::Mat &&
	        bitEnumContainsAll(closure(propertiesOf(ty), ty.getShape()),
		                   Properties::symmetric);
	    k.push_back(sym ? false : l.trans != transposed);
	    k.push_back(l.lower);
	}
	return k;
    }

    /// A term s (A B) of a sum with s known at run time (possibly with
    /// constant scales): the product with alpha and its run-time scalars.
    std::optional<
        std::tuple<MulOp, double, SmallVector<std::pair<Value, bool>>>>
    scaledProductParts(Value t)
    {
	double alpha = 1;
	SmallVector<std::pair<Value, bool>> scales;
	Value v = t;
	while (auto sc = v.getDefiningOp<ScaleOp>()) {
	    if (!sc->hasOneUse())
		return std::nullopt;
	    FloatAttr c;
	    if (matchPattern(sc.getAlpha(), m_Constant(&c)))
		alpha *= c.getValueAsDouble();
	    else
		scales.push_back({sc.getAlpha(), false});
	    v = sc.getOperand();
	}
	auto mul = v.getDefiningOp<MulOp>();
	if (scales.empty() || !mul || !v.hasOneUse())
	    return std::nullopt;
	return std::make_tuple(mul, alpha, scales);
    }

    template <typename P>
    std::optional<P>
    scaledProductAs(Value t)
    {
	auto parts = scaledProductParts(t);
	if (!parts)
	    return std::nullopt;
	auto &[m, a, sc] = *parts;
	return P{m, a, sc};
    }

    /// v (or an assume of it, or its transpose) is used outside the
    /// algebra: returned, yielded by a loop, passed to a call.
    static bool
    usedAsValue(Value v, bool throughTranspose, bool sumIsValue = false)
    {
	for (Operation *u : v.getUsers()) {
	    if (isa<AssumeOp>(u) || (throughTranspose && isa<TransposeOp>(u)) ||
	        (sumIsValue && isa<ScaleOp>(u))) {
		if (usedAsValue(u->getResult(0), throughTranspose, sumIsValue))
		    return true;
		continue;
	    }
	    if (!isUlmLamOp(u) || (sumIsValue && isa<AddOp>(u)))
		return true;
	}
	return false;
    }

    /// lstsq(A, B) without LAPACK: the normal equations (A^T A)^-1 A^T B,
    /// A^T A asserted spd (A has full column rank), then selected as any
    /// product (Cholesky, syrk).
    void
    normalEquations()
    {
	SmallVector<LstsqOp> ls;
	func.walk([&](LstsqOp op) { ls.push_back(op); });
	for (LstsqOp op : ls) {
	    b.setInsertionPoint(op);
	    Location loc = op.getLoc();
	    Value a = op.getA();
	    Value at = TransposeOp::create(b, loc, a).getResult();
	    Value g = MulOp::create(b, loc, ValueRange{at, a}).getResult();
	    auto gt = cast<RankedTensorType>(g.getType());
	    g = AssumeOp::create(
	        b, loc, withProperties(gt, propertiesOf(gt) | Properties::spd),
	        g, false);
	    Value gi = InverseOp::create(b, loc, g).getResult();
	    Value x = MulOp::create(b, loc, ValueRange{gi, at, op.getB()})
	                  .getResult();
	    if (x.getType() != op.getType())
		x = AssumeOp::create(b, loc, op.getType(), x, false);
	    op.getResult().replaceAllUsesWith(x);
	    op.erase();
	}
    }

    /// lstsq(A, B) with LAPACK: copies of op(A) and op(B), gels (QR) in
    /// place, the solution copied from the leading rows of B.
    Value
    selectLstsq(LstsqOp op)
    {
	Location loc = op.getLoc();
	auto leaf = [](Value v) {
	    while (auto as = v.getDefiningOp<AssumeOp>())
		v = as.getOperand();
	    Leaf l;
	    l.v = v;
	    if (auto t = v.getDefiningOp<TransposeOp>()) {
		l.v = t.getOperand();
		l.trans = true;
	    }
	    return l;
	};
	auto copyOf = [&](const Leaf &l) {
	    Value e = empty(l, l, elementType(l.v), loc);
	    return lk::GecopyOp::create(
	               b, loc, TypeRange{e.getType()},
	               l.trans ? lk::Trans::trans : lk::Trans::none, l.v, e)
	        ->getResult(0);
	};
	Leaf la = leaf(op.getA()), lb = leaf(op.getB());
	Value ac = copyOf(la), bc = copyOf(lb);
	auto gels = lk::GelsOp::create(
	    b, loc, TypeRange{ac.getType(), bc.getType(), b.getI64Type()}, ac,
	    bc);
	Leaf lat = la;
	lat.trans = !la.trans;
	Value x = empty(lat, lb, elementType(la.v), loc);
	return lk::GecopyOp::create(b, loc, TypeRange{x.getType()},
	                            lk::Trans::none, gels->getResult(1), x)
	    ->getResult(0);
    }

    /// The identity used as a value (a result, A^0): formed (gescal 0,
    /// adddiag 1).  In products it is dropped, in sums it is adddiag.
    void
    formIdentities()
    {
	SmallVector<IdentityOp> ids;
	func.walk([&](IdentityOp id) {
	    if (usedAsValue(id.getResult(), /*throughTranspose=*/true))
		ids.push_back(id);
	});
	for (IdentityOp id : ids) {
	    b.setInsertionPointAfter(id);
	    Location loc = id.getLoc();
	    auto t = cast<RankedTensorType>(stripProperties(id.getType()));
	    Value e = tensor::EmptyOp::create(b, loc, t.getShape(),
	                                      t.getElementType(),
	                                      ValueRange(id->getOperands()));
	    Type elt = t.getElementType();
	    e = lk::GescalOp::create(b, loc, TypeRange{e.getType()},
	                             constant(elt, 0.0, loc), e)
	            ->getResult(0);
	    e = lk::AdddiagOp::create(b, loc, TypeRange{e.getType()},
	                              constant(elt, 1.0, loc), e)
	            ->getResult(0);
	    if (e.getType() != id.getType())
		e = AssumeOp::create(b, loc, id.getType(), e, false);
	    SmallVector<OpOperand *> uses;
	    for (OpOperand &u : id->getUses())
		if (!isa<MulOp, AddOp, ScaleOp, InverseOp>(u.getOwner()))
		    uses.push_back(&u);
	    for (OpOperand *u : uses)
		u->set(e);
	}
    }

    // ---- the Woodbury identity ---------------------------------------------

    /// On a sum of the Woodbury form: distributed without a cost
    /// comparison.
    static constexpr llvm::StringLiteral kDistributeAttr = "ulm_lam.distribute";

    /// alpha I + beta Z^T Z (a sum of a scaled identity and a scaled Gram
    /// matrix, Z^T Z or X X^T = Z^T Z with Z = X^T): the scales and Z.
    struct LowRank
    {
	SmallVector<Value> alphas, betas;
	Value z;
    };

    static std::optional<LowRank>
    lowRankPlusIdentity(Value a)
    {
	while (auto as = a.getDefiningOp<AssumeOp>())
	    a = as.getOperand();
	auto add = a.getDefiningOp<AddOp>();
	if (!add || add.getTerms().size() != 2)
	    return std::nullopt;
	LowRank lr;
	bool id = false, gram = false;
	for (Value t : add.getTerms()) {
	    auto [scales, u] = unscaled(t);
	    if (isIdentity(u)) {
		lr.alphas = scales;
		id = true;
		continue;
	    }
	    while (auto as = u.getDefiningOp<AssumeOp>())
		u = as.getOperand();
	    auto m = u.getDefiningOp<MulOp>();
	    if (!m || m.getFactors().size() != 2)
		return std::nullopt;
	    Value f0 = m.getFactors()[0], f1 = m.getFactors()[1];
	    auto t0 = f0.getDefiningOp<TransposeOp>();
	    auto t1 = f1.getDefiningOp<TransposeOp>();
	    if (!(t0 && t0.getOperand() == f1) &&
	        !(t1 && t1.getOperand() == f0))
		return std::nullopt;
	    lr.z = f1; // Z^T Z: Z; X X^T: Z = X^T
	    lr.betas = scales;
	    gram = true;
	}
	if (!id || !gram)
	    return std::nullopt;
	return lr;
    }

    /// The product of scalars (1 for none), at b's insertion point.
    Value
    scalarProduct(ArrayRef<Value> xs, Type elt, Location loc)
    {
	Value r;
	for (Value x : xs)
	    r = r ? arith::MulFOp::create(b, loc, r, x).getResult() : x;
	return r ? r : constant(elt, 1.0, loc);
    }

    /// A tensor type of the shape rows(r) x cols(c) (static where known).
    static RankedTensorType
    shapeOf(Value r, unsigned rd, Value c, unsigned cd, Type elt)
    {
	auto rt = cast<RankedTensorType>(r.getType());
	auto ct = cast<RankedTensorType>(c.getType());
	return RankedTensorType::get({rt.getDimSize(rd), ct.getDimSize(cd)},
	                             elt);
    }

    /// The identity of size n x n with n = dimension d of v.
    Value
    identityLike(Value v, unsigned d, Type elt, Location loc)
    {
	RankedTensorType t = shapeOf(v, d, v, d, elt);
	SmallVector<Value> sizes;
	for (unsigned k = 0; k < 2; ++k)
	    if (t.isDynamicDim(k))
		sizes.push_back(dimValue(b, loc, v, d));
	return IdentityOp::create(
	           b, loc, TypeRange{withProperties(t, Properties::identity)},
	           sizes)
	    .getResult();
    }

    /// (alpha I + beta Z^T Z)^-1 = (1/alpha) (I - beta Z^T K^-1 Z) with
    /// K = alpha I + beta Z Z^T (the Woodbury identity): Z is N x p, the
    /// matrix to factorise is N x N instead of p x p.  K is spd if and only
    /// if alpha I + beta Z^T Z is (they share the eigenvalues alpha + beta
    /// s_i^2, and for N < p alpha is one of them).
    ///
    /// For an inverse used only inside products, the computation that
    /// depends on it, with the forming of the matrix itself, is generated
    /// twice, in an scf.if on N < p: with the Woodbury identity, and as
    /// written.  Then distribute() applies the products to the terms of
    /// the Woodbury form, so that no p x p matrix is formed: the predictive
    /// covariance Phi* A^-1 Phi*^T of Bayesian linear regression (the
    /// Laplace approximation of a last layer) needs, with N training points
    /// and p features, an N x N factorisation when N < p.
    void
    woodbury()
    {
	SmallVector<InverseOp> invs;
	func.walk([&](InverseOp i) { invs.push_back(i); });
	for (InverseOp inv : invs) {
	    std::optional<LowRank> lr = lowRankPlusIdentity(inv.getOperand());
	    // (used inside products, also scaled: beta A^-1 X^T t)
	    std::function<bool(Operation *)> inProducts = [&](Operation *u) {
		if (isa<MulOp>(u))
		    return true;
		return isa<ScaleOp>(u) && !u->use_empty() &&
		       llvm::all_of(u->getUsers(), inProducts);
	    };
	    if (!lr || inv->use_empty() ||
	        !llvm::all_of(inv->getUsers(), inProducts))
		continue;
	    Block *blk = inv->getBlock();
	    // the computation that depends on the inverse (ulm_lam ops of
	    // this block), and the ops that form the matrix, used by nothing
	    // else
	    llvm::SetVector<Operation *> slice;
	    slice.insert(inv);
	    bool ok = true;
	    for (size_t k = 0; k < slice.size() && ok; ++k)
		for (Value r : slice[k]->getResults())
		    for (Operation *u : r.getUsers()) {
			if (!isUlmLamOp(u))
			    continue; // a use outside: a result of the if
			if (u->getBlock() != blk) {
			    ok = false;
			    break;
			}
			slice.insert(u);
		    }
	    if (!ok)
		continue;
	    for (bool grown = true; grown;) {
		grown = false;
		for (size_t k = 0; k < slice.size(); ++k)
		    for (Value o : slice[k]->getOperands()) {
			Operation *d = o.getDefiningOp();
			if (!d || slice.contains(d) || d->getBlock() != blk ||
			    !isUlmLamOp(d) || o == lr->z)
			    continue;
			if (llvm::all_of(d->getUsers(), [&](Operation *u) {
			        return slice.contains(u);
			    })) {
			    slice.insert(d);
			    grown = true;
			}
		    }
	    }
	    SmallVector<Operation *> ops(slice.begin(), slice.end());
	    llvm::sort(ops, [](Operation *x, Operation *y) {
		return x->isBeforeInBlock(y);
	    });
	    Operation *last = ops.back();
	    // values used after the computation (results, calls, loops)
	    SmallVector<Value> outs;
	    for (Operation *op : ops)
		for (Value r : op->getResults())
		    for (Operation *u : r.getUsers())
			if (!slice.contains(u)) {
			    Operation *a = blk->findAncestorOpInBlock(*u);
			    if (!a || !last->isBeforeInBlock(a))
				ok = false;
			    if (!llvm::is_contained(outs, r))
				outs.push_back(r);
			}
	    // (the scalars must be defined before the computation)
	    for (Value x : llvm::concat<Value>(lr->alphas, lr->betas))
		if (Operation *d = x.getDefiningOp();
		    d && d->getBlock() == blk && !d->isBeforeInBlock(ops[0]))
		    ok = false;
	    if (!ok || outs.empty())
		continue;
	    Location loc = inv.getLoc();
	    Value z = lr->z;
	    Type elt = elementType(z);
	    b.setInsertionPointAfter(last);
	    Value n = dimValue(b, loc, z, 0), p = dimValue(b, loc, z, 1);
	    if (!n || !p)
		continue;
	    Value cond =
	        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, n, p);
	    SmallVector<Type> types;
	    for (Value r : outs)
		types.push_back(r.getType());
	    auto ifOp = scf::IfOp::create(b, loc, types, cond, true);
	    // as written
	    {
		b.setInsertionPointToStart(&ifOp.getElseRegion().front());
		IRMapping map;
		for (Operation *op : ops)
		    b.clone(*op, map);
		SmallVector<Value> ys;
		for (Value r : outs)
		    ys.push_back(map.lookup(r));
		scf::YieldOp::create(b, loc, ys);
	    }
	    // with the Woodbury identity
	    {
		b.setInsertionPointToStart(&ifOp.getThenRegion().front());
		Value alpha = scalarProduct(lr->alphas, elt, loc);
		Value beta = scalarProduct(lr->betas, elt, loc);
		Value zt = TransposeOp::create(b, loc, z).getResult();
		Value zzt =
		    MulOp::create(b, loc, ValueRange{z, zt}).getResult();
		Value in = identityLike(z, 0, elt, loc);
		auto nn = cast<RankedTensorType>(
		    stripProperties(shapeOf(z, 0, z, 0, elt)));
		Value k =
		    AddOp::create(
		        b, loc, TypeRange{nn},
		        ValueRange{
		            ScaleOp::create(b, loc, TypeRange{nn}, alpha, in)
		                .getResult(),
		            ScaleOp::create(b, loc, TypeRange{nn}, beta, zzt)
		                .getResult()})
		        .getResult();
		k = AssumeOp::create(
		    b, loc, withProperties(nn, Properties::spd), k, false);
		Value ki = InverseOp::create(b, loc, k).getResult();
		Value one = constant(elt, 1.0, loc);
		Value ia =
		    arith::DivFOp::create(b, loc, one, alpha).getResult();
		Value nb = arith::NegFOp::create(b, loc, beta).getResult();
		auto pp = cast<RankedTensorType>(
		    stripProperties(shapeOf(z, 1, z, 1, elt)));
		// a fresh (1/alpha)(I - beta Z^T K^-1 Z) for every use, so that
		// each product can distribute it (K^-1 is shared: one
		// factorisation)
		auto wFor = [&]() {
		    Value inner = MulOp::create(b, loc, ValueRange{zt, ki, z})
		                      .getResult();
		    Value t2 = ScaleOp::create(b, loc, TypeRange{pp}, nb, inner)
		                   .getResult();
		    Value ip = identityLike(z, 1, elt, loc);
		    auto sumOp = AddOp::create(b, loc, TypeRange{pp},
		                               ValueRange{ip, t2});
		    // (always distributed: the branch runs only for N < p)
		    sumOp->setAttr(kDistributeAttr, b.getUnitAttr());
		    Value sum = sumOp.getResult();
		    return ScaleOp::create(b, loc, TypeRange{pp}, ia, sum)
		        .getResult();
		};
		IRMapping map;
		for (Operation *op : ops) {
		    if (op == inv.getOperation())
			continue;
		    if (llvm::is_contained(op->getOperands(),
		                           inv.getResult())) {
			Operation *c = b.clone(*op, map);
			for (OpOperand &o : c->getOpOperands())
			    if (o.get() == inv.getResult()) {
				OpBuilder::InsertionGuard g(b);
				b.setInsertionPoint(c);
				Value w = wFor();
				o.set(w.getType() == o.get().getType()
				          ? w
					  : AssumeOp::create(b, loc,
					                     o.get().getType(),
					                     w, false)
					        .getResult());
			    }
			continue;
		    }
		    b.clone(*op, map);
		}
		SmallVector<Value> ys;
		for (Value r : outs)
		    ys.push_back(map.lookup(r));
		scf::YieldOp::create(b, loc, ys);
	    }
	    for (auto [k, r] : llvm::enumerate(outs))
		r.replaceUsesWithIf(ifOp.getResult(k), [&](OpOperand &o) {
		    return !slice.contains(o.getOwner()) &&
		           !ifOp->isAncestor(o.getOwner());
		});
	    for (Operation *op : llvm::reverse(ops))
		op->erase();
	}
    }

    // ---- distributivity -----------------------------------------------------

    /// The cost of a product by the generalised matrix chain at the current
    /// sample sizes (a sum among its factors counts as formed); kInf if it
    /// has no plan.
    /// Leaves for costing only: no factorisation is created (it would be
    /// on values not selected yet, whose types may still change).
    bool dryRun = false;

    /// The factorisation factorise would choose for m, standing for it.
    Factor
    kindOfFactor(Value m)
    {
	auto it = factors.find(m);
	if (it != factors.end())
	    return it->second;
	auto t = cast<RankedTensorType>(m.getType());
	Properties p = closure(propertiesOf(t), t.getShape());
	bool lo = bitEnumContainsAll(p, Properties::lower_tri),
	     up = bitEnumContainsAll(p, Properties::upper_tri);
	if (lo || up)
	    return {Factor::Triangular, m, {}, lo};
	if (bitEnumContainsAll(p, Properties::spd))
	    return {Factor::Cholesky, m, {}, true};
	return {Factor::LU, m, m, true,
	        lapack && bitEnumContainsAll(p, Properties::symmetric)};
    }

    double
    productCost(MulOp m)
    {
	llvm::SaveAndRestore<bool> dry(dryRun, true);
	Chain c;
	SmallVector<std::pair<Value, bool>> saved = std::move(dynScales);
	dynScales.clear();
	bool ok = true;
	for (Value f : m.getFactors())
	    if (failed(leavesOf(f, m, c.leaves, c.alpha))) {
		ok = false;
		break;
	    }
	dynScales = std::move(saved);
	if (!ok || c.leaves.empty())
	    return kInf;
	solveChain(c, m);
	return c.cost[0][c.leaves.size() - 1];
    }

    double
    sizeOf(Value v, unsigned d)
    {
	return sz(dimOf(v, d));
    }

    /// A term of a sum without its constant scales: (scales, the term).
    static std::pair<SmallVector<Value>, Value>
    unscaled(Value t)
    {
	SmallVector<Value> scales;
	while (auto sc = t.getDefiningOp<ScaleOp>()) {
	    scales.push_back(sc.getAlpha());
	    t = sc.getOperand();
	}
	return {scales, t};
    }

    /// An identity matrix (also under assume).
    static bool
    isIdentity(Value v)
    {
	while (auto as = v.getDefiningOp<AssumeOp>())
	    v = as.getOperand();
	return static_cast<bool>(v.getDefiningOp<IdentityOp>());
    }

    /// The cost of forming a term of a sum (a product, or an inverse as a
    /// solve with the identity), plus adding it.
    double
    termCost(Value t)
    {
	Value u = unscaled(t).second;
	double add = sizeOf(t, 0) * sizeOf(t, 1);
	if (auto m = u.getDefiningOp<MulOp>())
	    return productCost(m) + add;
	if (auto tr = u.getDefiningOp<TransposeOp>())
	    u = tr.getOperand();
	if (auto inv = u.getDefiningOp<InverseOp>()) {
	    auto ty = cast<RankedTensorType>(inv.getOperand().getType());
	    Properties p = closure(propertiesOf(ty), ty.getShape());
	    double n = sizeOf(u, 0);
	    bool tri = bitEnumContainsAll(p, Properties::lower_tri) ||
	               bitEnumContainsAll(p, Properties::upper_tri);
	    // (formInverses: potri, getri, or triangular solves with the
	    // identity; the factorisation is shared with solves)
	    double f = tri                                      ? 1.0
	               : bitEnumContainsAll(p, Properties::spd) ? 1.0 / 3.0
	                                                        : 4.0 / 3.0;
	    return f * n * n * n + add;
	}
	return add;
    }

    /// L (T_1 + ... + T_k) R = L T_1 R + ... + L T_k R when that is
    /// cheaper and a term is computed (never for a sum of given matrices,
    /// whose parentheses may be set for accuracy): a sum applied to a
    /// vector or a narrow matrix, as the
    /// adapted weight of LoRA, (W + B A) x = W x + B (A x), or a solve
    /// inside a sum, (C + B^-1) x = C x + B^-1 x, which then needs no
    /// explicit inverse.  Both sides are costed by the generalised matrix
    /// chain at the sample sizes of plansOf (all must agree, so with sizes
    /// known only at run time a distribution is never wrong by much); the
    /// sum must have no other use (otherwise it is formed anyway).
    void
    distribute()
    {
	SmallVector<MulOp> work;
	func.walk([&](MulOp m) { work.push_back(m); });
	while (!work.empty()) {
	    MulOp m = work.pop_back_val();
	    auto factors = llvm::to_vector(m.getFactors());
	    for (auto [i, f] : llvm::enumerate(factors)) {
		// (a scaled sum, s (T_1 + T_2): s scales every term)
		// (and through assume: its facts are about the sum, which is
		// not formed)
		SmallVector<Value> outer;
		Value u0 = f;
		while (Operation *d = u0.getDefiningOp()) {
		    if (!d->hasOneUse())
			break;
		    if (auto sc = dyn_cast<ScaleOp>(d)) {
			outer.push_back(sc.getAlpha());
			u0 = sc.getOperand();
		    } else if (auto as = dyn_cast<AssumeOp>(d)) {
			u0 = as.getOperand();
		    } else {
			break;
		    }
		}
		auto add = u0.getDefiningOp<AddOp>();
		if (!add || !add->hasOneUse())
		    continue;
		// an identity term is dropped from its product (L I R = L R;
		// alone, s I stays adddiag, already cheap)
		bool plain = factors.size() > 1 ||
		             llvm::none_of(add.getTerms(), [](Value t) {
		                 return isIdentity(unscaled(t).second);
		             });
		// a sum of given matrices, A (B - C) x, stays formed: its
		// parentheses may have been set for accuracy (B - C exact or
		// nearly so when B and C are close; A B x - A C x subtracts
		// rounded products).  Only a sum with a computed term (a
		// product, an inverse) is distributed.
		bool computedTerm = llvm::any_of(add.getTerms(), [](Value t) {
		    Value u = unscaled(t).second;
		    if (auto tr = u.getDefiningOp<TransposeOp>())
			u = tr.getOperand();
		    return u.getDefiningOp<MulOp>() ||
		           u.getDefiningOp<InverseOp>();
		});
		if (!plain || !computedTerm)
		    continue;
		// the distributed products, built before m (erased if not
		// cheaper)
		b.setInsertionPoint(m);
		Location loc = m.getLoc();
		auto rt = cast<RankedTensorType>(m.getType());
		Type plainTy = stripProperties(rt);
		SmallVector<Value> terms;
		SmallVector<Operation *> created;
		SmallVector<MulOp> products;
		for (Value t : add.getTerms()) {
		    auto [scales, u] = unscaled(t);
		    scales.append(outer.begin(), outer.end());
		    SmallVector<Value> fs(factors.begin(), factors.begin() + i);
		    if (auto um = u.getDefiningOp<MulOp>())
			llvm::append_range(fs, um.getFactors());
		    else if (!isIdentity(u))
			fs.push_back(u);
		    fs.append(factors.begin() + i + 1, factors.end());
		    Value v;
		    if (fs.size() == 1) { // L I = L
			v = fs[0];
		    } else {
			auto p = MulOp::create(b, loc, TypeRange{plainTy}, fs);
			created.push_back(p);
			products.push_back(p);
			v = p.getResult();
		    }
		    for (Value s : llvm::reverse(scales)) {
			auto sc =
			    ScaleOp::create(b, loc, TypeRange{plainTy}, s, v);
			created.push_back(sc);
			v = sc.getResult();
		    }
		    terms.push_back(v);
		}
		// the sizes known only at run time, sampled as in plansOf
		llvm::SetVector<Value> syms;
		auto collect = [&](Value v) {
		    for (unsigned d = 0; d < 2; ++d)
			if (auto s = dyn_cast_if_present<Value>(dimOf(v, d)))
			    syms.insert(s);
		};
		for (MulOp p : products)
		    for (Value g : p.getFactors())
			collect(g);
		bool forced = add->hasAttr(kDistributeAttr);
		bool cheaper = forced || syms.size() <= 4;
		unsigned masks = forced ? 0 : 1u << syms.size();
		for (unsigned mask = 0; cheaper && mask < masks; ++mask) {
		    sample.clear();
		    for (auto [k, v] : llvm::enumerate(syms))
			sample[v] = (mask >> k) & 1 ? 1000.0 : 10.0;
		    double keep = productCost(m);
		    for (Value t : add.getTerms())
			keep += termCost(t);
		    double dist = 0;
		    for (MulOp p : products)
			dist += productCost(p);
		    dist += add.getTerms().size() * sizeOf(m, 0) * sizeOf(m, 1);
		    cheaper = dist < keep;
		}
		sample.clear();
		if (!cheaper) {
		    for (Operation *op : llvm::reverse(created))
			op->erase();
		    continue;
		}
		Value sum = AddOp::create(b, loc, TypeRange{rt}, terms);
		m.getResult().replaceAllUsesWith(sum);
		m->erase();
		if (Operation *d = f.getDefiningOp(); d && d->use_empty())
		    eraseDead(d);
		for (MulOp p : products)
		    work.push_back(p);
		break;
	    }
	}
    }

    /// Erases op and the ulm_lam ops that only it used.
    static void
    eraseDead(Operation *op)
    {
	SmallVector<Value> operands(op->getOperands());
	op->erase();
	for (Value v : operands)
	    if (Operation *d = v.getDefiningOp();
	        d && v.use_empty() && isUlmLamOp(d) && d->use_empty())
		eraseDead(d);
    }

    /// The identities formed by formInverses: A^-1 = A^-1 E.
    llvm::SmallPtrSet<Value, 4> identityRhs;
    /// ... not needed by an explicit inverse (erased at the end of run).
    SmallVector<Value> unusedIdentities;

    /// An inverse used as a value (a result, a term of a sum, a loop's
    /// value), not as a factor of a product: inv(A) = inv(A) * E with E
    /// the identity, formed (gescal 0, adddiag 1), so that it is a solve
    /// with the identity (Cholesky, LU or triangular, as for any solve).
    void
    formInverses()
    {
	SmallVector<Value> targets;
	func.walk([&](InverseOp inv) {
	    // as a value, or a term of a sum (also scaled): formed
	    if (usedAsValue(inv.getResult(), false, true))
		targets.push_back(inv.getResult());
	    for (Operation *u : inv->getUsers())
		if (auto t = dyn_cast<TransposeOp>(u))
		    if (usedAsValue(t.getResult(), false, true))
			targets.push_back(t.getResult());
	});
	for (Value v : targets) {
	    // (inv(A))^T = (A^T)^-1: a transposed solve operand, as in products
	    SmallVector<OpOperand *> uses;
	    for (OpOperand &u : v.getUses())
		if (!isa<MulOp, InverseOp, TransposeOp>(u.getOwner()))
		    uses.push_back(&u);
	    b.setInsertionPointAfter(v.getDefiningOp());
	    Location loc = v.getLoc();
	    Value a = v.getDefiningOp()->getOperand(0);
	    if (auto inv = a.getDefiningOp<InverseOp>())
		a = inv.getOperand();
	    Leaf l;
	    l.v = a;
	    Type elt = elementType(l.v);
	    Value e = empty(l, l, elt, loc);
	    e = lk::GescalOp::create(b, loc, TypeRange{e.getType()},
	                             constant(elt, 0.0, loc), e)
	            ->getResult(0);
	    e = lk::AdddiagOp::create(b, loc, TypeRange{e.getType()},
	                              constant(elt, 1.0, loc), e)
	            ->getResult(0);
	    identityRhs.insert(e);
	    Value m =
	        MulOp::create(b, loc, TypeRange{v.getType()}, ValueRange{v, e})
	            .getResult();
	    for (OpOperand *u : uses)
		u->set(m);
	}
    }

    /// s * X with s known only at run time, used as a value (in a sum, a
    /// result, a loop): selected as gescal(s, X).  Inside products the
    /// scale is collected by leavesOf; s * I in a sum is adddiag.
    static bool
    dynamicScale(Operation *op)
    {
	auto sc = dyn_cast<ScaleOp>(op);
	if (!sc)
	    return false;
	// a constant scale is collected by the product or the sum it is in; one
	// of a value used outside the algebra (a result, a loop's value, an
	// argument of a call), e.g. A^-1 b * 2, is selected as gescal as well
	if (matchPattern(sc.getAlpha(), m_Constant()))
	    return usedAsValue(sc.getResult(), /*throughTranspose=*/false);
	Value v = sc.getOperand();
	while (true) {
	    if (auto s2 = v.getDefiningOp<ScaleOp>())
		v = s2.getOperand();
	    else if (auto as = v.getDefiningOp<AssumeOp>())
		v = as.getOperand();
	    else
		break;
	}
	if (v.getDefiningOp<IdentityOp>())
	    return false;
	// s (A B) in a sum: emitted by the sum (a gemm with alpha s)
	if (productInSum(op))
	    return false;
	return llvm::any_of(op->getUsers(), [](Operation *u) {
	    return !isa<MulOp, ScaleOp, TransposeOp, InverseOp>(u);
	});
    }

    /// A chain of scales (each used once) of a product (used once) that
    /// ends in a sum: s (A B) + ...
    static bool
    productInSum(Operation *scale)
    {
	Value v = scale->getResult(0);
	Operation *top = scale;
	while (top->hasOneUse() && isa<ScaleOp>(*top->getUsers().begin()))
	    top = *top->getUsers().begin();
	if (!top->hasOneUse() || !isa<AddOp>(*top->getUsers().begin()))
	    return false;
	v = cast<ScaleOp>(scale).getOperand();
	while (auto sc = v.getDefiningOp<ScaleOp>()) {
	    if (!sc->hasOneUse())
		return false;
	    v = sc.getOperand();
	}
	return v.getDefiningOp<MulOp>() && v.hasOneUse();
    }

    static bool
    feedsSum(Operation *op)
    {
	if (!isa<MulOp>(op) || !op->hasOneUse())
	    return false;
	// (through scales, constant or known at run time: s (A B) in a sum is
	// a gemm with alpha s)
	Operation *user = *op->getUsers().begin();
	while (isa<ScaleOp>(user) && user->hasOneUse())
	    user = *user->getUsers().begin();
	return isa<AddOp>(user);
    }

    static Type
    elementType(Value v)
    {
	return cast<RankedTensorType>(v.getType()).getElementType();
    }

    Value
    constant(Type elt, double v, Location loc)
    {
	return arith::ConstantOp::create(b, loc, elt, b.getFloatAttr(elt, v));
    }

    /// A new matrix with the rows of op(l) and the columns of op(r).
    Value
    empty(const Leaf &l, const Leaf &r, Type elt, Location loc)
    {
	auto t = RankedTensorType::get({l.rows(), r.cols()}, elt);
	FailureOr<SmallVector<Value>> sizes =
	    dynamicSizes(b, loc, t, l.v, l.trans ? 1 : 0, r.v, r.trans ? 0 : 1);
	assert(succeeded(sizes) && "dimensions checked by leavesOf");
	return tensor::EmptyOp::create(b, loc, t, *sizes);
    }

    /// `v` = alpha * inner for constant alpha (ulm_lam.scale), else alpha = 1.
    static std::pair<Value, double>
    peelScale(Value v)
    {
	if (auto s = v.getDefiningOp<ScaleOp>()) {
	    FloatAttr c;
	    if (matchPattern(s.getAlpha(), m_Constant(&c)))
		return {s.getOperand(), c.getValueAsDouble()};
	}
	return {v, 1.0};
    }

    // ---- storage -----------------------------------------------------------

    /// Kernel results of which only the lower triangle is computed (syrk).
    llvm::SmallPtrSet<Value, 4> lowerOnly;

    /// Scalars known at run time that scale factors of the product being
    /// collected (leavesOf), and whether they are inverted.
    SmallVector<std::pair<Value, bool>> dynScales;
    bool scalingProduct = false;
    /// chain() computes the transpose of the product: (F_1 ... F_k)^T =
    /// F_k^T ... F_1^T (a sum computed transposed, see selectAdd)
    bool transposedProduct = false;

    /// The product of the scalars (arith ops at the insertion point).
    Value
    scalarOf(ArrayRef<std::pair<Value, bool>> scales, Location loc)
    {
	Value s;
	for (auto [v, inverted] : scales) {
	    if (inverted) {
		Value one = arith::ConstantOp::create(
		    b, loc, v.getType(), b.getFloatAttr(v.getType(), 1.0));
		v = arith::DivFOp::create(b, loc, one, v);
	    }
	    s = s ? arith::MulFOp::create(b, loc, s, v).getResult() : v;
	}
	return s;
    }

    /// alpha * x in place, alpha a run-time value (one triangle stays so).
    Value
    scaleBy(Value alpha, Value x, Location loc)
    {
	// (one stored triangle stays so: gescal scales both, the stored one
	// correctly; also a value whose type says so, e.g. an inverse from
	// potri passed on by replace)
	int st = lowerOnly.contains(x) ? 1 : storageOf(x);
	if (st == 2)
	    x = full(x);
	Value r = lk::GescalOp::create(b, loc, TypeRange{x.getType()}, alpha, x)
	              ->getResult(0);
	if (st == 1)
	    lowerOnly.insert(r);
	return r;
    }

    /// Completed (full) versions of matrices stored as one triangle.
    llvm::DenseMap<Value, Value> completed;

    /// `m` (a symmetric matrix of which one triangle is stored) as a full
    /// matrix: ulm_lamk.sycomplete, once, right after the definition of m.
    /// In place: bufferization works on m itself if it may be overwritten
    /// and is not read afterwards, on a copy otherwise.
    Value
    full(Value m)
    {
	int st = storageOf(m);
	if (!st)
	    return m;
	if (Value c = completed.lookup(m))
	    return c;
	OpBuilder::InsertionGuard guard(b);
	if (Operation *def = m.getDefiningOp())
	    b.setInsertionPointAfter(def);
	else
	    b.setInsertionPointToStart(m.getParentBlock());
	Value c = lk::SycompleteOp::create(
	              b, m.getLoc(), TypeRange{m.getType()},
	              st == 1 ? lk::Uplo::lower : lk::Uplo::upper, m)
	              ->getResult(0);
	completed[m] = c;
	return c;
    }

    /// `l` as an operand of a kernel that reads the whole matrix.
    Leaf
    fullLeaf(Leaf l)
    {
	if (l.kind == Leaf::Mat && l.stored != Leaf::Full) {
	    l.v = full(l.v);
	    l.stored = Leaf::Full;
	}
	return l;
    }

    /// Which of two triangle-stored factors to complete: one that is
    /// complete already, else one that bufferization can complete in place
    /// (a computed matrix, or a parameter that may be overwritten), else
    /// the right one.
    bool
    completeLeft(const Leaf &l, const Leaf &r)
    {
	if (completed.contains(r.v))
	    return false;
	if (completed.contains(l.v))
	    return true;
	auto writable = [](Value v) {
	    auto a = dyn_cast<BlockArgument>(v);
	    if (!a)
		return true;
	    auto f = dyn_cast<func::FuncOp>(a.getOwner()->getParentOp());
	    return f && f.getArgAttr(a.getArgNumber(), "ulm_lam.inout");
	};
	return writable(l.v) && !writable(r.v);
    }

    // ---- factorisations ----------------------------------------------------

    /// Cost of factorising an n x n matrix with properties p (0: it is
    /// triangular, or already factorised).
    double
    factorCost(Value m)
    {
	if (factors.contains(m))
	    return 0;
	auto t = cast<RankedTensorType>(m.getType());
	Properties p = closure(propertiesOf(t), t.getShape());
	double n = costSize(t.getDimSize(0));
	if (bitEnumContainsAll(p, Properties::lower_tri) ||
	    bitEnumContainsAll(p, Properties::upper_tri))
	    return 0;
	if (bitEnumContainsAll(p, Properties::spd))
	    return n * n * n / 3;
	if (lapack && bitEnumContainsAll(p, Properties::symmetric))
	    return n * n * n / 3; // L D L^T
	return 2 * n * n * n / 3;
    }

    /// Invert the product m = F_1 ... F_k factor by factor
    /// (F_k^-1 ... F_1^-1) instead of forming and factorising it?  Possible
    /// if every factor is square and non-singular; chosen if factorising
    /// the distinct factors costs less than forming the product (unless
    /// it is needed anyway) and factorising it.  E.g. (R R)^-1 with R spd:
    /// one Cholesky factorisation of R (n^3/3) instead of R R (2n^3) and
    /// its factorisation (n^3/3); and cond(R) instead of cond(R)^2.
    bool
    invertFactorwise(MulOp m)
    {
	bool onlyInverted = llvm::all_of(
	    m->getUsers(), [](Operation *u) { return isa<InverseOp>(u); });
	if (m->use_empty() || !llvm::any_of(m->getUsers(), [](Operation *u) {
	        return isa<InverseOp>(u);
	    }))
	    return false;
	llvm::SmallPtrSet<Value, 4> distinct;
	double n = 0;
	for (Value f : m.getFactors()) {
	    Value v = f;
	    while (auto t = v.getDefiningOp<TransposeOp>())
		v = t.getOperand();
	    auto t = cast<RankedTensorType>(v.getType());
	    Properties p = closure(propertiesOf(t), t.getShape());
	    OpFoldResult r = dimOf(v, 0), c = dimOf(v, 1);
	    if (!r || r != c ||
	        !bitEnumContainsAll(p, Properties::non_singular))
		return false;
	    n = costSize(t.getDimSize(0));
	    distinct.insert(v);
	}
	double factorwise = 0;
	for (Value v : distinct)
	    factorwise += factorCost(v);
	double form =
	    onlyInverted ? 2 * n * n * n * (m.getFactors().size() - 1) : 0;
	return factorwise < form + factorCost(m.getResult());
    }

    /// Factorise `m` once, right after its definition.
    Factor
    factorise(Value m)
    {
	auto it = factors.find(m);
	if (it != factors.end())
	    return it->second;
	auto t = cast<RankedTensorType>(m.getType());
	Properties p = closure(propertiesOf(t), t.getShape());
	Factor f;
	int st = storageOf(m); // only one triangle of m is stored
	if (bitEnumContainsAll(p, Properties::lower_tri) ||
	    bitEnumContainsAll(p, Properties::upper_tri)) {
	    f = {Factor::Triangular,
	         m,
	         {},
	         st ? st == 1 : bitEnumContainsAll(p, Properties::lower_tri)};
	} else {
	    OpBuilder::InsertionGuard guard(b);
	    if (Operation *def = m.getDefiningOp())
		b.setInsertionPointAfter(def);
	    else
		b.setInsertionPointToStart(m.getParentBlock());
	    Location loc = m.getLoc();
	    if (bitEnumContainsAll(p, Properties::spd)) {
		// A = L L^T, or A = U^T U from the upper triangle
		bool lower = st != 2;
		auto potrf = lk::PotrfOp::create(
		    b, loc, TypeRange{m.getType(), b.getI64Type()},
		    lower ? lk::Uplo::lower : lk::Uplo::upper, m);
		f = {Factor::Cholesky, potrf->getResult(0), {}, lower};
	    } else {
		if (st)
		    m = full(m); // LU reads the whole matrix
		auto pivTy = RankedTensorType::get(
		    {t.getDimSize(0)}, IntegerType::get(b.getContext(), 32));
		SmallVector<Value> n;
		if (pivTy.isDynamicDim(0))
		    n.push_back(dimValue(b, loc, m, 0));
		Value piv = tensor::EmptyOp::create(b, loc, pivTy, n);
		auto getrf = lk::GetrfOp::create(
		    b, loc, TypeRange{m.getType(), pivTy, b.getI64Type()}, m,
		    piv);
		// symmetric, not known to be spd: L D L^T (half the flops)
		bool sym =
		    lapack && bitEnumContainsAll(p, Properties::symmetric);
		if (sym)
		    getrf.setSymmetric(true);
		f = {Factor::LU, getrf->getResult(0), getrf->getResult(1), true,
		     sym};
	    }
	}
	factors[m] = f;
	return f;
    }

    /// The leaves of one factor of a product.
    /// (trans, inv: the factor is v^T, v^-1 or v^-T.)
    LogicalResult
    leavesOf(Value v, Operation *user, SmallVectorImpl<Leaf> &out,
             double &alpha, bool trans = false, bool inv = false)
    {
	while (true) {
	    if (auto t = v.getDefiningOp<TransposeOp>()) {
		trans = !trans;
		v = t.getOperand();
	    } else if (auto i = v.getDefiningOp<InverseOp>()) {
		inv = !inv;
		v = i.getOperand();
	    } else if (auto sc = v.getDefiningOp<ScaleOp>()) {
		// a constant factor moves out of the product: (a A)^T = a A^T,
		// (a A)^-1 = a^-1 A^-1
		FloatAttr c;
		if (!matchPattern(sc.getAlpha(), m_Constant(&c))) {
		    // a scalar known at run time: the product is scaled
		    // afterwards (gescal; 1/s inside an inverse)
		    dynScales.push_back({sc.getAlpha(), inv});
		    v = sc.getOperand();
		    continue;
		}
		double a = c.getValueAsDouble();
		if (inv && a == 0)
		    return user->emitError("ulm-lam-select-kernels: inverse of "
		                           "a matrix scaled by 0");
		alpha *= inv ? 1.0 / a : a;
		v = sc.getOperand();
	    } else {
		break;
	    }
	}
	auto t = cast<RankedTensorType>(v.getType());
	for (unsigned d = 0; d < 2; ++d)
	    if (!dimOf(v, d))
		return user->emitError(
		           "ulm-lam-select-kernels: unknown size of "
		           "a dynamic dimension (missing ")
		       << kDimsAttr << "?)";
	if (!inv) {
	    Leaf l{Leaf::Mat, v, {}, trans, true};
	    if (int st = storageOf(v)) {
		l.stored = st == 1 ? Leaf::Lower : Leaf::Upper;
		l.trans = false; // symmetric
	    }
	    out.push_back(l);
	    return success();
	}
	// (F_1 ... F_k)^-1 = F_k^-1 ... F_1^-1, (F_1 ... F_k)^-T = F_1^-T ...
	// F_k^-T: factorise the factors instead of the product
	if (auto m = v.getDefiningOp<MulOp>(); m && expanded.contains(m)) {
	    SmallVector<Value> fs(m.getFactors());
	    if (!trans)
		std::reverse(fs.begin(), fs.end());
	    for (Value f : fs)
		if (failed(leavesOf(f, user, out, alpha, trans, true)))
		    return failure();
	    return success();
	}
	// (costing only, see distribute: the kind of factorisation without
	// creating it; the leaves stand for v)
	Factor f = dryRun ? kindOfFactor(v) : factorise(v);
	switch (f.kind) {
	case Factor::Triangular:
	    out.push_back({Leaf::TriSolve, f.f, {}, trans, f.lower});
	    break;
	case Factor::Cholesky:
	    // (L L^T)^-1 = L^-T L^-1 and (U^T U)^-1 = U^-1 U^-T, also for the
	    // transpose
	    out.push_back({Leaf::TriSolve, f.f, {}, f.lower, f.lower});
	    out.push_back({Leaf::TriSolve, f.f, {}, !f.lower, f.lower});
	    break;
	case Factor::LU: {
	    Leaf l{Leaf::LUSolve, f.f, f.piv, trans, true};
	    l.sym = f.sym;
	    out.push_back(l);
	    break;
	}
	}
	return success();
    }

    // ---- the generalised matrix chain --------------------------------------

    /// Sizes assumed for dynamic dimensions while comparing costs (a
    /// sample of the run-time sizes; kNominal for the others).
    llvm::DenseMap<Value, double> sample;
    /// Emitting one of several plans inside a branch: its results do not
    /// dominate later products (no reuse), and the product must be
    /// complete (no syrk).
    bool inBranch = false;

    double
    sz(OpFoldResult d) const
    {
	if (auto a = dyn_cast_if_present<Attribute>(d))
	    return static_cast<double>(cast<IntegerAttr>(a).getInt());
	if (auto v = dyn_cast_if_present<Value>(d)) {
	    auto it = sample.find(v);
	    return it != sample.end() ? it->second : kNominal;
	}
	return kNominal;
    }

    struct Chain
    {
	SmallVector<Leaf> leaves;
	double alpha = 1.0;     // constant factors moved out of the product
	bool symmetric = false; // the product is symmetric (F P F^T)
	std::vector<std::vector<double>> cost;
	std::vector<std::vector<Choice>> choice;
    };

    static bool
    isVector(int64_t cols)
    {
	return cols == 1;
    }

    /// Interval [i, j] can be the in-place operand of a solve: it is computed
    /// into a buffer, or it is a single matrix.  A transposed one (a
    /// transposed leaf, or a computed interval reused transposed) is first
    /// copied transposed (gecopy, see transposedOperand).
    static bool
    solveOperand(const Chain &c, int i, int j)
    {
	if (i < j)
	    return c.cost[i][j] < kInf;
	return c.leaves[i].kind == Leaf::Mat;
    }

    /// The operand [i, j] of a solve is a transposed view: it costs a
    /// transposed copy (rows x cols of the interval).
    static bool
    transposedOperand(const Chain &c, int i, int j)
    {
	if (i < j)
	    return c.choice[i][j].kind == Choice::Reuse && c.choice[i][j].trans;
	return c.leaves[i].trans;
    }

    /// op(l) as an untransposed matrix: a transposed one is copied.
    Leaf
    untransposed(Leaf l, Location loc)
    {
	if (!l.trans)
	    return l;
	Value e = empty(l, l, elementType(l.v), loc);
	l.v = lk::GecopyOp::create(b, loc, TypeRange{e.getType()},
	                           lk::Trans::trans, l.v, e)
	          ->getResult(0);
	l.trans = false;
	return l;
    }

    void
    solveChain(Chain &c, Operation *user)
    {
	int k = c.leaves.size();
	c.cost.assign(k, std::vector<double>(k, kInf));
	c.choice.assign(k, std::vector<Choice>(k));
	auto &L = c.leaves;
	for (int i = 0; i < k; ++i)
	    c.cost[i][i] = L[i].kind == Leaf::Mat ? 0 : kInf;
	auto consider = [&](int i, int j, double cost, Choice ch) {
	    if (cost <= c.cost[i][j]) { // ties: the later split, i.e. (A B) C
		c.cost[i][j] = cost;
		c.choice[i][j] = ch;
	    }
	};
	for (int len = 2; len <= k; ++len)
	    for (int i = 0; i + len - 1 < k; ++i) {
		int j = i + len - 1;
		double m = sz(L[i].rowsDim()), n = sz(L[j].colsDim());
		// computed by an earlier product (not the whole product: that
		// would be a common subexpression, which CSE removes)
		if (len < k) {
		    for (bool tr : {false, true}) {
			auto it = computed.find(keyOf(L, i, j, tr));
			if (it != computed.end() &&
			    !unreusable.contains(it->second) &&
			    dom.properlyDominates(it->second, user)) {
			    c.cost[i][j] = 0;
			    c.choice[i][j] = {Choice::Reuse, 0, it->second, tr};
			    break;
			}
		    }
		    if (c.cost[i][j] == 0)
			continue;
		}
		for (int s = i; s < j; ++s) {
		    double inner = sz(L[s].colsDim());
		    // gemm / gemv of two computable parts
		    if (c.cost[i][s] < kInf && c.cost[s + 1][j] < kInf)
			consider(i, j,
			         c.cost[i][s] + c.cost[s + 1][j] +
			             2 * m * inner * n,
			         {Choice::Gemm, s});
		    // a solve applied from the left to the right part (in
		    // place)
		    if (s == i && L[i].isSolve() && solveOperand(c, s + 1, j)) {
			double f = L[i].kind == Leaf::LUSolve ? 2 : 1;
			double copy =
			    transposedOperand(c, s + 1, j) ? m * n : 0;
			consider(i, j, c.cost[s + 1][j] + f * m * m * n + copy,
			         {Choice::SolveLeft, s});
		    }
		    // a triangular solve applied from the right to the left
		    // part
		    // (with LU: getrs on the transposed view, side(right))
		    if (s + 1 == j && L[j].isSolve() && solveOperand(c, i, s))
			consider(i, j,
			         c.cost[i][s] +
			             (L[j].kind == Leaf::LUSolve ? 2 : 1) * m *
			                 n * n +
			             (transposedOperand(c, i, s) ? m * n : 0),
			         {Choice::SolveRight, s});
		}
		// Z^T Z: the left half is the transpose of the right half
		if (len % 2 == 0) {
		    int h = i + len / 2;
		    bool mirror = true;
		    for (int t = 0; t < len / 2 && mirror; ++t)
			mirror = L[i + t].isTransposeOf(L[j - t]);
		    if (mirror && c.cost[h][j] < kInf) {
			double zr = sz(L[h].rowsDim());
			// syrk: one triangle, n^2 zr (as a gemm where the
			// result must be complete, see emitKernels)
			double cost = c.cost[h][j] + n * zr * n;
			if (cost < c.cost[i][j]) {
			    c.cost[i][j] = cost;
			    c.choice[i][j] = {Choice::Gram, h};
			}
		    }
		}
	    }
    }

    /// Interval [i, j] as a kernel operand: a single matrix leaf as it is
    /// (possibly transposed), otherwise computed into a new buffer.
    Leaf
    operand(Chain &c, int i, int j, Location loc)
    {
	if (i == j)
	    return c.leaves[i];
	if (c.choice[i][j].kind == Choice::Reuse)
	    return {Leaf::Mat,
	            c.choice[i][j].value,
	            {},
	            c.choice[i][j].trans,
	            true};
	Leaf r{Leaf::Mat, emit(c, i, j, {}, 1, 0, loc), {}, false, true};
	if (lowerOnly.contains(r.v)) // a Gram matrix, one triangle (syrk)
	    r.stored = Leaf::Lower;
	return r;
    }

    Value
    solveInPlace(const Leaf &s, Value rhs, Location loc)
    {
	Type elt = elementType(rhs);
	int64_t n = cast<RankedTensorType>(rhs.getType()).getDimSize(1);
	auto trans = s.trans ? lk::Trans::trans : lk::Trans::none;
	if (s.kind == Leaf::LUSolve) {
	    auto g =
	        lk::GetrsOp::create(b, loc, TypeRange{rhs.getType()},
		                    lk::Side::left, trans, s.v, s.piv, rhs);
	    if (s.sym)
		g.setSymmetric(true);
	    return g->getResult(0);
	}
	auto uplo = s.lower ? lk::Uplo::lower : lk::Uplo::upper;
	if (isVector(n))
	    return lk::TrsvOp::create(b, loc, TypeRange{rhs.getType()}, uplo,
	                              trans, lk::Diag::non_unit, s.v, rhs)
	        ->getResult(0);
	return lk::TrsmOp::create(
	           b, loc, TypeRange{rhs.getType()}, lk::Side::left, uplo,
	           trans, lk::Diag::non_unit, constant(elt, 1, loc), s.v, rhs)
	    ->getResult(0);
    }

    /// x copied into dest (bufferization.materialize_in_destination, a
    /// memref.copy after bufferization); the result is dest's buffer.
    Value
    into(Value x, Value dest, Location loc)
    {
	if (x.getType() != dest.getType())
	    x = AssumeOp::create(b, loc, dest.getType(), x, false);
	return bufferization::MaterializeInDestinationOp::create(b, loc, x,
	                                                         dest)
	    .getResult();
    }

    /// x := x op(T)^-1 in place, T triangular or LU factors (a solve from
    /// the right).
    Value
    solveRightInPlace(const Leaf &t, Value x, Location loc)
    {
	if (t.kind == Leaf::LUSolve) {
	    auto g = lk::GetrsOp::create(
	        b, loc, TypeRange{x.getType()}, lk::Side::right,
	        t.trans ? lk::Trans::trans : lk::Trans::none, t.v, t.piv, x);
	    if (t.sym)
		g.setSymmetric(true);
	    return g->getResult(0);
	}
	return lk::TrsmOp::create(
	           b, loc, TypeRange{x.getType()}, lk::Side::right,
	           t.lower ? lk::Uplo::lower : lk::Uplo::upper,
	           t.trans ? lk::Trans::trans : lk::Trans::none,
	           lk::Diag::non_unit, constant(elementType(x), 1, loc), t.v, x)
	    ->getResult(0);
    }

    /// alpha * product of leaves [i, j] computed in the new destination
    /// dest itself (a tensor.empty): a gemm writes into it, a chain that
    /// ends with solves copies its first operand into it and solves there
    /// in place.  So results of different branches (scf.if) that are
    /// computed into the same destination share its buffer.
    Value
    emitInto(Chain &c, int i, int j, Value dest, double alpha, Location loc)
    {
	Choice ch = c.choice[i][j];
	int s = ch.split;
	auto scale = [&](Value x) -> Value {
	    if (alpha == 1)
		return x;
	    return lk::GescalOp::create(b, loc, TypeRange{x.getType()},
	                                constant(elementType(x), alpha, loc), x)
	        ->getResult(0);
	};
	// interval [a, z] in dest: computed into it, or copied
	auto part = [&](int a, int z) -> Value {
	    if (a < z && c.choice[a][z].kind != Choice::Reuse)
		return emitInto(c, a, z, dest, 1.0, loc);
	    Leaf l = fullLeaf(operand(c, a, z, loc));
	    if (l.trans) // copied transposed into dest
		return lk::GecopyOp::create(b, loc, TypeRange{dest.getType()},
		                            lk::Trans::trans, l.v, dest)
		    ->getResult(0);
	    return into(l.v, dest, loc);
	};
	switch (ch.kind) {
	case Choice::Gemm:
	case Choice::Gram:
	    return emit(c, i, j, dest, alpha, 0.0, loc);
	case Choice::SolveLeft:
	    return scale(solveInPlace(c.leaves[i], part(s + 1, j), loc));
	case Choice::SolveRight:
	    return scale(solveRightInPlace(c.leaves[j], part(i, s), loc));
	default:
	    return accumulate(emit(c, i, j, {}, 1, 0, loc), dest, alpha, 0.0,
	                      loc);
	}
    }

    /// dest = beta dest + alpha x for kernels that cannot accumulate.
    Value
    accumulate(Value x, Value dest, double alpha, double beta, Location loc)
    {
	Type elt = elementType(x);
	if (!dest) {
	    if (alpha == 1)
		return x;
	    return lk::GescalOp::create(b, loc, TypeRange{x.getType()},
	                                constant(elt, alpha, loc), x)
	        ->getResult(0);
	}
	// a new destination: copy (scaling it by beta = 0 would read
	// uninitialised memory, 0 * NaN = NaN)
	if (beta == 0 && dest.getDefiningOp<tensor::EmptyOp>()) {
	    Value r = into(x, dest, loc);
	    if (alpha == 1)
		return r;
	    return lk::GescalOp::create(b, loc, TypeRange{r.getType()},
	                                constant(elt, alpha, loc), r)
	        ->getResult(0);
	}
	// one triangle plus one triangle stays one triangle; one triangle plus
	// a whole matrix: completed first
	bool lower = lowerOnly.contains(dest) && lowerOnly.contains(x);
	if (lowerOnly.contains(dest) && !lower)
	    dest = completeNow(dest, loc);
	if (lowerOnly.contains(x) && !lower)
	    x = completeNow(x, loc);
	if (beta != 1)
	    dest = lk::GescalOp::create(b, loc, TypeRange{dest.getType()},
	                                constant(elt, beta, loc), dest)
	               ->getResult(0);
	Value r = lk::GeaxpyOp::create(b, loc, TypeRange{dest.getType()},
	                               constant(elt, alpha, loc), x, dest)
	              ->getResult(0);
	if (lower)
	    lowerOnly.insert(r);
	return r;
    }

    /// A value of which only the lower triangle is computed, completed here
    /// (sycomplete at b's insertion point).
    Value
    completeNow(Value v, Location loc)
    {
	return lk::SycompleteOp::create(b, loc, TypeRange{v.getType()},
	                                lk::Uplo::lower, v)
	    ->getResult(0);
    }

    Value
    gemm(Leaf l, Leaf r, Value dest, double alpha, double beta, Location loc,
         Value alphaV = {})
    {
	Type elt = elementType(l.v);
	// alpha, or a run-time value (a scaled product, see chain)
	auto alphaOf = [&] {
	    return alphaV ? alphaV : constant(elt, alpha, loc);
	};
	if (!dest)
	    dest = empty(l, r, elt, loc);
	// (an accumulator of which only the lower triangle is computed)
	if (beta != 0 && lowerOnly.contains(dest))
	    dest = completeNow(dest, loc);
	auto t = [](bool x) { return x ? lk::Trans::trans : lk::Trans::none; };
	// a symmetric factor stored as one triangle: symm / symv (if both
	// are, one of them is completed first)
	if (l.stored != Leaf::Full && r.stored != Leaf::Full) {
	    if (completeLeft(l, r))
		l = fullLeaf(l);
	    else
		r = fullLeaf(r);
	}
	if (l.stored != Leaf::Full || r.stored != Leaf::Full) {
	    bool left = l.stored != Leaf::Full;
	    const Leaf &a = left ? l : r, &g = left ? r : l;
	    auto uplo =
	        a.stored == Leaf::Lower ? lk::Uplo::lower : lk::Uplo::upper;
	    if (left && isVector(g.cols()) && !g.trans)
		return lk::SymvOp::create(b, loc, TypeRange{dest.getType()},
		                          uplo, alphaOf(), a.v, g.v,
		                          constant(elt, beta, loc), dest)
		    ->getResult(0);
	    return lk::SymmOp::create(b, loc, TypeRange{dest.getType()},
	                              left ? lk::Side::left : lk::Side::right,
	                              uplo, t(g.trans), alphaOf(), a.v, g.v,
	                              constant(elt, beta, loc), dest)
	        ->getResult(0);
	}
	// a triangular factor (lower_tri / upper_tri): the other factor copied
	// into dest, then trmm in place (half the flops of gemm); trmm cannot
	// accumulate, so only for beta = 0
	if (beta == 0) {
	    auto tri = [&](const Leaf &x) -> int {
		if (x.kind != Leaf::Mat || x.stored != Leaf::Full)
		    return 0;
		auto ty = cast<RankedTensorType>(x.v.getType());
		Properties p = closure(propertiesOf(ty), ty.getShape());
		if (bitEnumContainsAll(p, Properties::lower_tri))
		    return 1;
		if (bitEnumContainsAll(p, Properties::upper_tri))
		    return 2;
		return 0;
	    };
	    int tl = tri(l), tr = tri(r);
	    if (tl || tr) {
		bool left = tl != 0;
		const Leaf &a = left ? l : r, &g = left ? r : l;
		auto ty = cast<RankedTensorType>(a.v.getType());
		bool unit =
		    bitEnumContainsAll(closure(propertiesOf(ty), ty.getShape()),
		                       Properties::unit_diag);
		Value d =
		    lk::GecopyOp::create(b, loc, TypeRange{dest.getType()},
		                         t(g.trans), g.v, dest)
		        ->getResult(0);
		return lk::TrmmOp::create(
		           b, loc, TypeRange{d.getType()},
		           left ? lk::Side::left : lk::Side::right,
		           (left ? tl : tr) == 1 ? lk::Uplo::lower
			                         : lk::Uplo::upper,
		           t(a.trans),
		           unit ? lk::Diag::unit : lk::Diag::non_unit,
		           alphaOf(), a.v, d)
		    ->getResult(0);
	    }
	}
	if (isVector(r.cols()) && !r.trans)
	    return lk::GemvOp::create(b, loc, TypeRange{dest.getType()},
	                              t(l.trans), alphaOf(), l.v, r.v,
	                              constant(elt, beta, loc), dest)
	        ->getResult(0);
	return lk::GemmOp::create(b, loc, TypeRange{dest.getType()}, t(l.trans),
	                          t(r.trans), alphaOf(), l.v, r.v,
	                          constant(elt, beta, loc), dest)
	    ->getResult(0);
    }

    /// Emit the product of leaves [i, j] as chosen by the DP; the result is
    /// beta * dest + alpha * product (dest may be null: beta = 0).
    Value
    emit(Chain &c, int i, int j, Value dest, double alpha, double beta,
         Location loc)
    {
	Value r = emitKernels(c, i, j, dest, alpha, beta, loc);
	// a plain product: remember it for later products (not one triangle:
	// later products read the whole matrix)
	if (alpha == 1 && beta == 0 && !lowerOnly.contains(r) && !inBranch &&
	    (!dest || dest.getDefiningOp<tensor::EmptyOp>()))
	    computed[keyOf(c.leaves, i, j, false)] = r;
	return r;
    }

    /// An accumulator syrk may update: symmetric, its lower triangle
    /// valid (stored completely or as the lower triangle).
    static bool
    symmetricAcc(Value v)
    {
	auto t = cast<RankedTensorType>(v.getType());
	return bitEnumContainsAll(closure(propertiesOf(t), t.getShape()),
	                          Properties::symmetric) &&
	       storageOf(v) != 2;
    }

    Value
    emitKernels(Chain &c, int i, int j, Value dest, double alpha, double beta,
                Location loc)
    {
	Choice ch = c.choice[i][j];
	int s = ch.split;
	switch (ch.kind) {
	case Choice::Gemm: {
	    Leaf l = operand(c, i, s, loc), r = operand(c, s + 1, j, loc);
	    // the last product of a symmetric chain (F P F^T): one triangle
	    // (gemmtr, half the flops), passed on as stored in the lower
	    // triangle, as a syrk result; into a new matrix or added to a
	    // symmetric one
	    bool top = i == 0 && j + 1 == static_cast<int>(c.leaves.size());
	    bool fresh =
	        beta == 0 && (!dest || dest.getDefiningOp<tensor::EmptyOp>());
	    bool update = beta != 0 && dest && symmetricAcc(dest);
	    if (top && c.symmetric && (!inBranch || noSandwich) &&
	        l.kind == Leaf::Mat && r.kind == Leaf::Mat &&
	        l.stored == Leaf::Full && r.stored == Leaf::Full &&
	        !isVector(r.cols()) && (fresh || update)) {
		Type elt = elementType(l.v);
		if (!dest)
		    dest = empty(l, r, elt, loc);
		auto t = [](bool x) {
		    return x ? lk::Trans::trans : lk::Trans::none;
		};
		Value g =
		    lk::GemmtrOp::create(
		        b, loc, TypeRange{dest.getType()}, lk::Uplo::lower,
		        t(l.trans), t(r.trans), constant(elt, alpha, loc), l.v,
		        r.v, constant(elt, beta, loc), dest)
		        ->getResult(0);
		lowerOnly.insert(g);
		return g;
	    }
	    return gemm(l, r, dest, alpha, beta, loc);
	}
	case Choice::Gram: {
	    Leaf z = operand(c, s, j, loc);
	    // the whole product into a new matrix, or added to a symmetric
	    // one (C = beta C + alpha Z^T Z, e.g. K** - V^T V): syrk computes
	    // one triangle (half the flops of gemm); the result is passed on
	    // as stored in its lower triangle (lower_stored, see replace)
	    // (also a part of a longer product, A A^T B: its one triangle is
	    // read by symm, or completed for kernels that read the whole
	    // matrix; see operand)
	    bool top = i == 0 && j + 1 == static_cast<int>(c.leaves.size());
	    bool fresh =
	        beta == 0 && (!dest || dest.getDefiningOp<tensor::EmptyOp>());
	    bool update = beta != 0 && dest &&
	                  (symmetricAcc(dest) || lowerOnly.contains(dest));
	    if (!inBranch && z.stored == Leaf::Full &&
	        (fresh || (top && update))) {
		Type elt = elementType(z.v);
		if (!dest) {
		    Leaf zt = z;
		    zt.trans = !z.trans;
		    dest = empty(zt, z, elt, loc);
		}
		// C = alpha op(A) op(A)^T with op(A) = Z^T
		Value r = lk::SyrkOp::create(b, loc, TypeRange{dest.getType()},
		                             lk::Uplo::lower,
		                             z.trans ? lk::Trans::none
		                                     : lk::Trans::trans,
		                             constant(elt, alpha, loc), z.v,
		                             constant(elt, beta, loc), dest)
		              ->getResult(0);
		lowerOnly.insert(r);
		return r;
	    }
	    Leaf zt = z;
	    zt.trans = !z.trans;
	    if (dest && beta != 0 && lowerOnly.contains(dest))
		dest = completeNow(dest, loc); // gemm reads the whole matrix
	    return gemm(zt, z, dest, alpha, beta, loc);
	}
	case Choice::SolveLeft: {
	    Leaf rhs = untransposed(fullLeaf(operand(c, s + 1, j, loc)), loc);
	    return accumulate(solveInPlace(c.leaves[i], rhs.v, loc), dest,
	                      alpha, beta, loc);
	}
	case Choice::SolveRight: {
	    Leaf lhs = untransposed(fullLeaf(operand(c, i, s, loc)), loc);
	    Value x = solveRightInPlace(c.leaves[j], lhs.v, loc);
	    return accumulate(x, dest, alpha, beta, loc);
	}
	case Choice::Reuse: // only proper subintervals, used via operand()
	case Choice::None:
	    break;
	}
	llvm_unreachable("interval without a kernel sequence");
    }

    using Plan = std::vector<std::vector<Choice>>;

    static bool
    samePlan(const Plan &a, const Plan &b)
    {
	return a.empty() || sameFrom(a, b, 0, a.size() - 1);
    }

    /// The same kernels for interval [i, j]: only the intervals the plan
    /// reaches from there count (the DP fills the whole table, and two
    /// size samples may differ in an interval no plan uses, which would
    /// give a decision at run time between identical branches).
    static bool
    sameFrom(const Plan &a, const Plan &b, int i, int j)
    {
	const Choice &x = a[i][j], &y = b[i][j];
	if (x.kind != y.kind || x.split != y.split || x.value != y.value ||
	    x.trans != y.trans)
	    return false;
	if (i == j)
	    return true;
	int s = x.split;
	switch (x.kind) {
	case Choice::Gemm:
	    return sameFrom(a, b, i, s) && sameFrom(a, b, s + 1, j);
	case Choice::SolveLeft:
	    return sameFrom(a, b, s + 1, j);
	case Choice::SolveRight:
	    return sameFrom(a, b, i, s);
	case Choice::Gram:
	    return sameFrom(a, b, s, j);
	default:
	    return true;
	}
    }

    /// The plans that are optimal for some sizes: the DP for every
    /// assignment of {10, 1000} to the (at most 4) distinct dynamic sizes
    /// of the chain.  Without dynamic sizes: the one plan.
    SmallVector<Plan>
    plansOf(Chain &c, Operation *user)
    {
	llvm::SetVector<Value> syms;
	for (const Leaf &l : c.leaves)
	    for (OpFoldResult d : {l.rowsDim(), l.colsDim()})
		if (auto v = dyn_cast_if_present<Value>(d))
		    syms.insert(v);
	int k = c.leaves.size();
	SmallVector<Plan> plans;
	sample.clear();
	if (syms.empty() || syms.size() > 4) {
	    solveChain(c, user);
	    if (c.cost[0][k - 1] < kInf)
		plans.push_back(c.choice);
	    return plans;
	}
	for (unsigned mask = 0; mask < (1u << syms.size()); ++mask) {
	    for (auto [t, v] : llvm::enumerate(syms))
		sample[v] = (mask >> t) & 1 ? 1000.0 : 10.0;
	    solveChain(c, user);
	    if (c.cost[0][k - 1] == kInf)
		continue;
	    if (llvm::none_of(plans, [&](const Plan &p) {
		    return samePlan(p, c.choice);
	        }))
		plans.push_back(c.choice);
	}
	sample.clear();
	return plans;
    }

    /// An index value for a dimension.
    Value
    dimIndex(OpFoldResult d, Location loc)
    {
	if (auto a = dyn_cast<Attribute>(d))
	    return arith::ConstantIndexOp::create(
	        b, loc, cast<IntegerAttr>(a).getInt());
	return cast<Value>(d);
    }

    /// The cost of interval [i, j] of plan c.choice from the run-time
    /// sizes (index arithmetic, as the DP computes it).
    Value
    costValue(Chain &c, int i, int j, Location loc)
    {
	auto cst = [&](int64_t v) {
	    return arith::ConstantIndexOp::create(b, loc, v).getResult();
	};
	auto mul = [&](Value x, Value y) {
	    return arith::MulIOp::create(b, loc, x, y).getResult();
	};
	auto add = [&](Value x, Value y) {
	    return arith::AddIOp::create(b, loc, x, y).getResult();
	};
	if (i == j)
	    return cst(0);
	const Choice &ch = c.choice[i][j];
	auto &L = c.leaves;
	Value m = dimIndex(L[i].rowsDim(), loc);
	Value n = dimIndex(L[j].colsDim(), loc);
	int s = ch.split;
	switch (ch.kind) {
	case Choice::Gemm:
	    return add(
	        add(costValue(c, i, s, loc), costValue(c, s + 1, j, loc)),
	        mul(cst(2), mul(mul(m, dimIndex(L[s].colsDim(), loc)), n)));
	case Choice::SolveLeft: {
	    Value v = add(costValue(c, s + 1, j, loc),
	                  mul(cst(L[i].kind == Leaf::LUSolve ? 2 : 1),
	                      mul(mul(m, m), n)));
	    return transposedOperand(c, s + 1, j) ? add(v, mul(m, n)) : v;
	}
	case Choice::SolveRight: {
	    Value v = add(costValue(c, i, s, loc),
	                  mul(cst(L[j].kind == Leaf::LUSolve ? 2 : 1),
	                      mul(mul(m, n), n)));
	    return transposedOperand(c, i, s) ? add(v, mul(m, n)) : v;
	}
	case Choice::Gram:
	    return add(costValue(c, s, j, loc),
	               mul(mul(n, dimIndex(L[s].rowsDim(), loc)), n));
	case Choice::Reuse:
	case Choice::None:
	    break;
	}
	return cst(0);
    }

    /// The product a selected value was computed from (K = P H^T S^-1): a
    /// later product with K as a factor can expand it (see chain).
    llvm::DenseMap<Value, MulOp> origin;

    // in the other branch of choleskySandwich (its result is passed on as
    // one triangle, so gemmtr is allowed there)
    bool noSandwich = false;

    /// A S A^T with S spd (F P F^T in a Kalman filter): S = L L^T (potrf),
    /// W = A L (a copy of A, trmm in place), C = alpha W W^T + beta C
    /// (syrk, one triangle): n^3/3 + m n^2 + m^2 n flops instead of
    /// 2 m n^2 + m^2 n (gemm, gemmtr), for S n x n and A m x n.  potrf
    /// runs at a lower rate than gemm for small n: its efficiency is
    /// modelled as n / (n + n0), which makes the Cholesky path cheaper when
    /// n + n0 < 3 m (n0 = 2400 fits OpenBLAS on an Apple M5: F P F^T with
    /// n = 500, 1000, 2000, 4000 takes 1.3, 1.04, 0.83, 0.79 times as long;
    /// a measured cost model is future work).  Decided when the code is
    /// generated if the sizes are known, otherwise at run time (scf.if, the
    /// usual plan in the other branch).
    std::optional<FailureOr<Value>>
    choleskySandwich(Chain &c, MulOp mul, Value dest, double alpha, double beta)
    {
	if (c.leaves.size() != 3 || !c.symmetric)
	    return std::nullopt;
	const Leaf &a = c.leaves[0], &s = c.leaves[1], &at = c.leaves[2];
	if (a.kind != Leaf::Mat || s.kind != Leaf::Mat ||
	    at.kind != Leaf::Mat || a.v != at.v || a.trans == at.trans ||
	    s.stored != Leaf::Full || a.stored != Leaf::Full)
	    return std::nullopt;
	auto st = cast<RankedTensorType>(s.v.getType());
	if (!bitEnumContainsAll(closure(propertiesOf(st), st.getShape()),
	                        Properties::spd))
	    return std::nullopt;
	bool fresh =
	    beta == 0 && (!dest || dest.getDefiningOp<tensor::EmptyOp>());
	bool update = beta != 0 && dest && symmetricAcc(dest);
	if (!fresh && !update)
	    return std::nullopt;
	auto aty = cast<RankedTensorType>(a.v.getType());
	int64_t n = st.getDimSize(0), m = aty.getDimSize(a.trans ? 1 : 0);
	bool known = !ShapedType::isDynamic(n) && !ShapedType::isDynamic(m);
	constexpr int64_t n0 = 2400;
	if (known && !(n + n0 < 3 * m))
	    return std::nullopt;
	Location loc = mul.getLoc();
	if (!dest)
	    b.setInsertionPoint(mul);
	Type elt = elementType(s.v);
	auto t = [](bool x) { return x ? lk::Trans::trans : lk::Trans::none; };
	auto chol = [&](Value d) -> Value {
	    // factorised in a copy (S may be a loop's state, or used after)
	    Value sc = empty(s, s, elt, loc);
	    sc = lk::GecopyOp::create(b, loc, TypeRange{sc.getType()},
	                              lk::Trans::none, s.v, sc)
	             ->getResult(0);
	    auto potrf = lk::PotrfOp::create(
	        b, loc, TypeRange{sc.getType(), b.getI64Type()},
	        lk::Uplo::lower, sc);
	    Value l = potrf->getResult(0);
	    Value w = empty(a, s, elt, loc);
	    w = lk::GecopyOp::create(b, loc, TypeRange{w.getType()}, t(a.trans),
	                             a.v, w)
	            ->getResult(0);
	    w = lk::TrmmOp::create(b, loc, TypeRange{w.getType()},
	                           lk::Side::right, lk::Uplo::lower,
	                           lk::Trans::none, lk::Diag::non_unit,
	                           constant(elt, 1.0, loc), l, w)
	            ->getResult(0);
	    if (!d) {
		Leaf wl;
		wl.v = w;
		Leaf wt = wl;
		wt.trans = true;
		d = empty(wl, wt, elt, loc);
	    }
	    Value r = lk::SyrkOp::create(b, loc, TypeRange{d.getType()},
	                                 lk::Uplo::lower, lk::Trans::none,
	                                 constant(elt, alpha * c.alpha, loc), w,
	                                 constant(elt, beta, loc), d)
	                  ->getResult(0);
	    lowerOnly.insert(r);
	    return r;
	};
	if (known)
	    return FailureOr<Value>(chol(dest));
	// the sizes are known at run time only: n < 3 m decides
	Value nv = dimValue(b, loc, s.v, 0);
	Value mv = dimValue(b, loc, a.v, a.trans ? 1 : 0);
	Value three = arith::ConstantIndexOp::create(b, loc, 3);
	Value cond = arith::CmpIOp::create(
	    b, loc, arith::CmpIPredicate::ult,
	    arith::AddIOp::create(b, loc, nv,
	                          arith::ConstantIndexOp::create(b, loc, n0)),
	    arith::MulIOp::create(b, loc, mv, three));
	// one buffer for both branches (each computes into it in place): a
	// given destination is copied into it first
	Value d;
	if (dest) {
	    Leaf dl;
	    dl.v = dest;
	    Value e = empty(dl, dl, elt, loc);
	    d = lk::GecopyOp::create(b, loc, TypeRange{e.getType()},
	                             lk::Trans::none, dest, e)
	            ->getResult(0);
	    // the copy has the properties of dest (symmetric: gemmtr may
	    // accumulate onto it)
	    if (d.getType() != dest.getType())
		d = AssumeOp::create(b, loc, dest.getType(), d, false);
	} else {
	    d = *newDest(mul);
	}
	Type resTy = d.getType();
	auto ifOp = scf::IfOp::create(b, loc, TypeRange{resTy}, cond,
	                              /*withElse=*/true);
	bool saved = inBranch;
	inBranch = true;
	{
	    OpBuilder::InsertionGuard g(b);
	    b.setInsertionPointToStart(ifOp.thenBlock());
	    Value r = chol(d);
	    if (r.getType() != resTy)
		r = AssumeOp::create(b, loc, resTy, r, false);
	    scf::YieldOp::create(b, loc, r);
	    b.setInsertionPointToStart(ifOp.elseBlock());
	    noSandwich = true;
	    FailureOr<Value> o = chain(mul, d, alpha, dest ? beta : 0.0);
	    noSandwich = false;
	    if (failed(o)) {
		inBranch = saved;
		return o;
	    }
	    Value ov = *o;
	    if (ov.getType() != resTy)
		ov = AssumeOp::create(b, loc, resTy, ov, false);
	    scf::YieldOp::create(b, loc, ov);
	}
	inBranch = saved;
	b.setInsertionPointAfter(ifOp);
	lowerOnly.insert(ifOp.getResult(0));
	return FailureOr<Value>(ifOp.getResult(0));
    }

    /// A factor that is itself a computed product (K in K H P, with
    /// K = P H^T S^-1) may be replaced by the leaves of that product, if
    /// the longer chain is cheaper: it can expose structure across the two
    /// products (here (H P)^T S^-1 (H P): trsm and syrk instead of gemm),
    /// and intervals already computed (H P, the factorisation of S, L^-1 H P)
    /// are reused.  Found by the equality saturation experiment
    /// (experiments/egraph): the e-graph keeps both forms of the product.
    /// The cost of product mul with its factor fi as it is (first) and
    /// with fi replaced by the factors of the product it was computed by
    /// (second), the expanded chain in *expandedOut.
    std::optional<std::pair<double, double>>
    expansionCosts(MulOp mul, unsigned fi, MulOp om, Chain *expandedOut,
                   bool symmetric)
    {
	auto factors = mul.getFactors();
	Chain c1, c2;
	c1.symmetric = c2.symmetric = symmetric;
	SmallVector<std::pair<Value, bool>> saved = std::move(dynScales);
	dynScales.clear();
	bool ok = true;
	for (unsigned j = 0; j < factors.size() && ok; ++j) {
	    ok = succeeded(leavesOf(factors[j], mul, c1.leaves, c1.alpha));
	    if (j != fi) {
		ok = ok &&
		     succeeded(leavesOf(factors[j], mul, c2.leaves, c2.alpha));
		continue;
	    }
	    for (Value g : om.getFactors())
		ok = ok && succeeded(leavesOf(g, mul, c2.leaves, c2.alpha));
	}
	ok = ok && dynScales.empty();
	dynScales = std::move(saved);
	if (!ok)
	    return std::nullopt;
	int k1 = c1.leaves.size(), k2 = c2.leaves.size();
	solveChain(c1, mul);
	solveChain(c2, mul);
	std::pair<double, double> r{c1.cost[0][k1 - 1], c2.cost[0][k2 - 1]};
	if (expandedOut)
	    *expandedOut = std::move(c2);
	return r;
    }

    /// The cost of computing the product om (its own chain).
    double
    formingCost(MulOp om)
    {
	Chain c;
	SmallVector<std::pair<Value, bool>> saved = std::move(dynScales);
	dynScales.clear();
	bool ok = true;
	for (Value g : om.getFactors())
	    ok = ok && succeeded(leavesOf(g, om, c.leaves, c.alpha));
	ok = ok && dynScales.empty();
	dynScales = std::move(saved);
	if (!ok || c.leaves.empty())
	    return 0;
	solveChain(c, om);
	return c.cost[0][c.leaves.size() - 1];
    }

    /// A factor that is itself a computed product (K in K H P, with
    /// K = P H^T S^-1) may be replaced by the leaves of that product, if
    /// the longer chain is cheaper: it can expose structure across the two
    /// products (here (H P)^T S^-1 (H P): trsm and syrk instead of gemm),
    /// and intervals already computed (H P, the factorisation of S, L^-1 H P)
    /// are reused.  Found by the equality saturation experiment
    /// (experiments/egraph): the e-graph keeps both forms of the product.
    ///
    /// If every other use of the factor is expanded as well, the factor
    /// itself is never needed: then the cost of forming it counts for the
    /// expansion (K r in the Kalman update, x + K r, with K H P expanded:
    /// (H P)^T L^-T L^-1 r, two trsv and a gemv instead of the trsm that
    /// forms K; found by experiments/egraph/lamp_ir.py).
    void
    expandFactors(Chain &c, MulOp mul)
    {
	auto factors = mul.getFactors();
	for (unsigned fi = 0; fi < factors.size(); ++fi) {
	    Value f = factors[fi];
	    while (auto as = f.getDefiningOp<AssumeOp>())
		f = as.getOperand();
	    auto it = origin.find(f);
	    if (it == origin.end())
		continue;
	    MulOp om = it->second;
	    Chain c2;
	    auto costs = expansionCosts(mul, fi, om, &c2, c.symmetric);
	    if (!costs)
		continue;
	    auto [cost1, cost2] = *costs;
	    if (!(cost2 < cost1) && otherUsesExpand(f, mul)) {
		// f never formed: the expanded chain without reusing it
		unreusable.insert(f);
		Chain c3;
		auto costs3 = expansionCosts(mul, fi, om, &c3, c.symmetric);
		if (costs3 && costs3->second < cost1 + formingCost(om)) {
		    c = std::move(c3);
		    continue;
		}
		unreusable.erase(f);
	    }
	    if (cost2 < cost1)
		c = std::move(c2);
	}
    }

    /// Are all uses of the computed product f other than `mul` factors of
    /// products that expand it (by the cost comparison alone)?
    bool
    otherUsesExpand(Value f, MulOp mul)
    {
	auto it = origin.find(f);
	if (it == origin.end())
	    return false;
	MulOp om = it->second;
	SmallVector<Value> vals{f};
	bool any = false;
	for (size_t i = 0; i < vals.size(); ++i)
	    for (OpOperand &u : vals[i].getUses()) {
		Operation *o = u.getOwner();
		if (auto as = dyn_cast<AssumeOp>(o)) {
		    vals.push_back(as.getResult());
		    continue;
		}
		auto um = dyn_cast<MulOp>(o);
		if (!um)
		    return false;
		if (um == mul)
		    continue;
		auto t = cast<RankedTensorType>(um.getType());
		bool sym =
		    bitEnumContainsAll(closure(propertiesOf(t), t.getShape()),
		                       Properties::symmetric);
		auto costs =
		    expansionCosts(um, u.getOperandNumber(), om, nullptr, sym);
		if (!costs || !(costs->second < costs->first))
		    return false;
		any = true;
	    }
	return any;
    }

    /// A^-1 E with E an identity of formInverses (an inverse used as a
    /// value) formed explicitly from the factorisation of A, shared with
    /// the solves with A: potri (Cholesky; the other triangle completed),
    /// 2 n^3 / 3 flops with potrf, or getri (LU), 2 n^3 with getrf; solving
    /// with the identity costs 7 n^3 / 3 and 8 n^3 / 3.  (Triangular
    /// matrices, L D L^T and transposed LU factors: solved with E.)
    std::optional<Value>
    explicitInverse(Chain &c, Location loc)
    {
	auto &L = c.leaves;
	if (L.empty() || L.back().kind != Leaf::Mat ||
	    !identityRhs.contains(L.back().v))
	    return std::nullopt;
	if (L.size() == 3 && L[0].kind == Leaf::TriSolve &&
	    L[1].kind == Leaf::TriSolve && L[0].v == L[1].v &&
	    L[0].trans != L[1].trans) {
	    // only a Cholesky factor appears twice, as L^-T L^-1
	    auto uplo = L[0].lower ? lk::Uplo::lower : lk::Uplo::upper;
	    Value x = lk::PotriOp::create(b, loc, TypeRange{L[0].v.getType()},
	                                  uplo, L[0].v)
	                  ->getResult(0);
	    // the lower triangle is passed on as such (completed where a use
	    // needs the whole matrix, see replace)
	    if (L[0].lower) {
		lowerOnly.insert(x);
		return x;
	    }
	    return lk::SycompleteOp::create(b, loc, TypeRange{x.getType()},
	                                    uplo, x)
	        ->getResult(0);
	}
	if (L.size() == 2 && L[0].kind == Leaf::LUSolve && !L[0].sym &&
	    !L[0].trans)
	    return lk::GetriOp::create(b, loc, TypeRange{L[0].v.getType()},
	                               L[0].piv, L[0].v)
	        ->getResult(0);
	return std::nullopt;
    }

    /// beta * dest + alpha * (product), dest null: a new value.
    FailureOr<Value>
    chain(MulOp mul, Value dest, double alpha, double beta,
          ArrayRef<std::pair<Value, bool>> extraScales = {})
    {
	Chain c;
	{
	    auto t = cast<RankedTensorType>(mul.getType());
	    c.symmetric = bitEnumContainsAll(
	        closure(propertiesOf(t), t.getShape()), Properties::symmetric);
	}
	dynScales.clear();
	{
	    SmallVector<Value> fs(mul.getFactors());
	    if (transposedProduct)
		std::reverse(fs.begin(), fs.end());
	    for (Value f : fs)
		if (failed(
		        leavesOf(f, mul, c.leaves, c.alpha, transposedProduct)))
		    return failure();
	}
	dynScales.append(extraScales.begin(), extraScales.end());
	if (dynScales.empty() && expand)
	    expandFactors(c, mul);
	if (dynScales.empty() && !inBranch && !noSandwich)
	    if (auto r = choleskySandwich(c, mul, dest, alpha, beta))
		return *r;
	if (!dynScales.empty() && !scalingProduct) {
	    SmallVector<std::pair<Value, bool>> scales = std::move(dynScales);
	    // a single plan whose last kernel is a gemm (gemv): the scalar is
	    // its alpha, beta * dest + (s alpha) A B in one kernel, as an
	    // expert writes W - step X^T R
	    SmallVector<Plan> plans = plansOf(c, mul);
	    int k = c.leaves.size();
	    if (plans.size() == 1 && plans[0][0][k - 1].kind == Choice::Gemm &&
	        !inBranch) {
		c.choice = plans[0];
		int s = c.choice[0][k - 1].split;
		Location loc = mul.getLoc();
		if (!dest)
		    b.setInsertionPoint(mul);
		Leaf l = operand(c, 0, s, loc),
		     r = operand(c, s + 1, k - 1, loc);
		Value sv = scalarOf(scales, loc);
		Type elt = elementType(l.v);
		Value av = arith::MulFOp::create(
		    b, loc, constant(elt, alpha * c.alpha, loc), sv);
		Value d = dest ? dest : *newDest(mul);
		return gemm(l, r, d, alpha * c.alpha, dest ? beta : 0.0, loc,
		            av);
	    }
	    // likewise a Gram matrix: one syrk, s alpha Z^T Z added to a
	    // symmetric accumulator (A <- rho A + c a^T a with c known at run
	    // time: one triangle, completed where needed, see replace)
	    if (plans.size() == 1 && plans[0][0][k - 1].kind == Choice::Gram &&
	        !inBranch &&
	        (!dest || symmetricAcc(dest) || lowerOnly.contains(dest))) {
		c.choice = plans[0];
		int h = c.choice[0][k - 1].split;
		Location loc = mul.getLoc();
		if (!dest)
		    b.setInsertionPoint(mul);
		Leaf z = operand(c, h, k - 1, loc);
		if (z.stored == Leaf::Full) {
		    Type elt = elementType(z.v);
		    Value sv = scalarOf(scales, loc);
		    Value av = arith::MulFOp::create(
		        b, loc, constant(elt, alpha * c.alpha, loc), sv);
		    Value d = dest;
		    if (!d) {
			Leaf zt = z;
			zt.trans = !z.trans;
			d = empty(zt, z, elt, loc);
		    }
		    Value r =
		        lk::SyrkOp::create(
		            b, loc, TypeRange{d.getType()}, lk::Uplo::lower,
		            z.trans ? lk::Trans::none : lk::Trans::trans, av,
		            z.v, constant(elt, dest ? beta : 0.0, loc), d)
		            ->getResult(0);
		    lowerOnly.insert(r);
		    return r;
		}
	    }
	    // otherwise: the product without s, then scaled, then added
	    // (the product is emitted at its own position; scale and sum at
	    // the caller's, where the accumulator is defined)
	    OpBuilder::InsertPoint ip = b.saveInsertionPoint();
	    scalingProduct = true;
	    FailureOr<Value> r = chain(mul, Value(), alpha, 0.0);
	    scalingProduct = false;
	    if (dest)
		b.restoreInsertionPoint(ip);
	    if (failed(r))
		return failure();
	    // (a scalar defined after the product, e.g. 1/a in A^-1 x / a,
	    // or the product itself emitted after the accumulator's position,
	    // e.g. a choice at run time: the scaling goes after it)
	    SmallVector<Value> after{*r};
	    for (auto &sc : scales)
		after.push_back(sc.first);
	    for (Value x : after) {
		Operation *d = x.getDefiningOp();
		Block *blk = b.getInsertionBlock();
		if (!d || d->getBlock() != blk)
		    continue;
		auto ipt = b.getInsertionPoint();
		if (ipt != blk->end() && !d->isBeforeInBlock(&*ipt))
		    b.setInsertionPointAfter(d);
	    }
	    Value x = scaleBy(scalarOf(scales, mul.getLoc()), *r, mul.getLoc());
	    if (!dest)
		return x;
	    // (one triangle, e.g. of a syrk, added to a whole matrix: completed
	    // first)
	    if (lowerOnly.contains(x))
		x = lk::SycompleteOp::create(b, mul.getLoc(),
		                             TypeRange{x.getType()},
		                             lk::Uplo::lower, x)
		        ->getResult(0);
	    return accumulate(x, dest, 1.0, beta, mul.getLoc());
	}
	if (dynScales.empty()) {
	    if (!dest)
		b.setInsertionPoint(mul);
	    if (std::optional<Value> x = explicitInverse(c, mul.getLoc())) {
		Value r = *x;
		unusedIdentities.push_back(c.leaves.back().v);
		double a = alpha * c.alpha;
		// one triangle (potri): a value as it is (replace bridges the
		// types), completed before it is added to a whole matrix
		if (lowerOnly.contains(r) && !dest && a == 1)
		    return r;
		if (lowerOnly.contains(r))
		    r = lk::SycompleteOp::create(b, mul.getLoc(),
		                                 TypeRange{r.getType()},
		                                 lk::Uplo::lower, r)
		            ->getResult(0);
		if (r.getType() != mul.getType())
		    r = AssumeOp::create(b, mul.getLoc(), mul.getType(), r,
		                         false);
		return accumulate(r, dest, a, beta, mul.getLoc());
	    }
	}
	SmallVector<Plan> plans = plansOf(c, mul);
	int k = c.leaves.size();
	if (plans.empty())
	    return mul.emitError(
	               "ulm-lam-select-kernels: no kernel sequence for this "
	               "product (an explicit inverse would be needed)"),
	           failure();
	// at the product, or with an accumulator at the sum (the insertion
	// point of selectAdd): the product may have been moved out of a loop
	// in which the accumulator is defined (X + A B with X loop-carried)
	if (!dest)
	    b.setInsertionPoint(mul);
	if (plans.size() > 1)
	    return choosePlan(c, plans, mul, dest, alpha, beta);
	if (!dest && !inBranch)
	    if (std::optional<Value> r =
	            hoistOperator(c, plans.front(), mul, alpha, beta))
		return *r;
	c.choice = plans.front();
	Choice top = c.choice[0][k - 1];
	if (!dest && (top.kind == Choice::Gemm || top.kind == Choice::Gram)) {
	    auto t = cast<RankedTensorType>(mul.getType());
	    FailureOr<SmallVector<Value>> sizes = dynamicSizes(
	        b, mul.getLoc(), t, mul.getResult(), 0, mul.getResult(), 1);
	    if (failed(sizes))
		return mul.emitError("ulm-lam-select-kernels: unknown size of "
		                     "the product");
	    dest = tensor::EmptyOp::create(b, mul.getLoc(), t.getShape(),
	                                   t.getElementType(), *sizes,
	                                   t.getEncoding());
	}
	return emit(c, 0, k - 1, dest, alpha * c.alpha, beta, mul.getLoc());
    }

    /// A new matrix for the whole product.
    FailureOr<Value>
    newDest(MulOp mul)
    {
	auto t = cast<RankedTensorType>(mul.getType());
	FailureOr<SmallVector<Value>> sizes = dynamicSizes(
	    b, mul.getLoc(), t, mul.getResult(), 0, mul.getResult(), 1);
	if (failed(sizes))
	    return mul.emitError("ulm-lam-select-kernels: unknown size of the "
	                         "product"),
	           failure();
	return tensor::EmptyOp::create(b, mul.getLoc(), t.getShape(),
	                               t.getElementType(), *sizes,
	                               t.getEncoding())
	    .getResult();
    }

    /// Several plans, each optimal for some sizes: compute their costs
    /// from the run-time sizes and run the cheapest one,
    ///
    ///   if (cost_0 <= cost_1 && cost_0 <= cost_2) { plan 0 }
    ///   else if (cost_1 <= cost_2) { plan 1 } else { plan 2 }
    ///
    /// (scf.if, yielding the product).
    FailureOr<Value>
    choosePlan(Chain &c, ArrayRef<Plan> plans, MulOp mul, Value dest,
               double alpha, double beta)
    {
	Location loc = mul.getLoc();
	int k = c.leaves.size();
	SmallVector<Value> costs;
	for (const Plan &p : plans) {
	    c.choice = p;
	    costs.push_back(costValue(c, 0, k - 1, loc));
	}
	Type resultTy = mul.getType();
	// All plans compute into the same destination (a new one, or the
	// accumulator), so that the branches yield one buffer: otherwise
	// bufferization gives each branch its own allocation and the scf.if
	// result an ownership known only at run time.
	Value shared = dest ? Value() : *newDest(mul);
	auto emitPlan = [&](size_t p) -> Value {
	    c.choice = plans[p];
	    Value r = shared
	                  ? emitInto(c, 0, k - 1, shared, alpha * c.alpha, loc)
	                  : emit(c, 0, k - 1, dest, alpha * c.alpha, beta, loc);
	    if (r.getType() != resultTy)
		r = AssumeOp::create(b, loc, resultTy, r, false);
	    return r;
	};
	bool saved = inBranch;
	inBranch = true;
	std::function<Value(size_t)> build = [&](size_t p) -> Value {
	    if (p + 1 == plans.size())
		return emitPlan(p);
	    Value cond;
	    for (size_t q = p + 1; q < plans.size(); ++q) {
		Value le = arith::CmpIOp::create(
		    b, loc, arith::CmpIPredicate::ule, costs[p], costs[q]);
		cond = cond
		           ? arith::AndIOp::create(b, loc, cond, le).getResult()
		           : le;
	    }
	    auto ifOp = scf::IfOp::create(b, loc, TypeRange{resultTy}, cond,
	                                  /*withElse=*/true);
	    {
		OpBuilder::InsertionGuard g(b);
		b.setInsertionPointToStart(ifOp.thenBlock());
		scf::YieldOp::create(b, loc, emitPlan(p));
		b.setInsertionPointToStart(ifOp.elseBlock());
		scf::YieldOp::create(b, loc, build(p + 1));
	    }
	    return ifOp.getResult(0);
	};
	Value r = build(0);
	inBranch = saved;
	return r;
    }

    /// A product in a loop whose factors are loop invariant except the
    /// last one, e.g. b_i = S^-1 H^T R^-1 y_i: instead of the whole chain
    /// in every step (O(n^2) per step: solves with the factors of R), the
    /// operator P = S^-1 H^T R^-1 can be formed once before the loop
    /// (O(n^2 m)) and applied in every step (gemv, O(n m)).  That pays off
    /// if the loop runs often enough:
    ///
    ///   cost(P) + trips * 2 rows(P) cols(P) cols(y) < trips * cost(step)
    ///
    /// The trip count is usually known only at run time, so the generated
    /// code decides: the condition and P (if chosen) before the loop, and
    /// in the loop both variants under the same condition (a constant trip
    /// count folds it).  Returns nothing if the product has no such form.
    /// Flops of a matrix-vector kernel in units of matrix-matrix flops
    /// (see hoistOperator).
    static constexpr int64_t kLevel2 = 10;

    std::optional<Value>
    hoistOperator(Chain &c, const Plan &plan, MulOp mul, double alpha,
                  double beta)
    {
	auto loop = mul->getParentOfType<scf::ForOp>();
	int k = c.leaves.size();
	if (!loop || k < 3 || beta != 0)
	    return std::nullopt;
	auto outside = [&](Value v) {
	    return !v || loop.isDefinedOutsideOfLoop(v);
	};
	const Leaf &last = c.leaves[k - 1];
	if (last.kind != Leaf::Mat || outside(last.v))
	    return std::nullopt;
	for (int i = 0; i + 1 < k; ++i)
	    if (!outside(c.leaves[i].v) || !outside(c.leaves[i].piv))
		return std::nullopt;
	for (const Leaf &l : c.leaves)
	    for (OpFoldResult d : {l.rowsDim(), l.colsDim()})
		if (!outside(dyn_cast_if_present<Value>(d)))
		    return std::nullopt;
	// the operator: the invariant prefix as a chain of its own, its
	// reused parts must be available before the loop
	Chain pre;
	pre.leaves.assign(c.leaves.begin(), c.leaves.end() - 1);
	SmallVector<Plan> prePlans = plansOf(pre, loop);
	// or its transpose: P^T = F_k^T ... F_1^T, e.g. (S^-1 H^T R^-1)^T =
	// R^-1 H S^-1 = L^-T (L^-1 H) S^-1, where the reused L^-1 H can be
	// solved with in place (H^T L^-T cannot: a transposed view)
	bool transposed = false;
	if (prePlans.empty()) {
	    Chain t;
	    for (auto it = pre.leaves.rbegin(); it != pre.leaves.rend(); ++it) {
		Leaf l = *it;
		if (l.stored == Leaf::Full)
		    l.trans = !l.trans;
		t.leaves.push_back(l);
	    }
	    prePlans = plansOf(t, loop);
	    pre = std::move(t);
	    transposed = true;
	}
	if (prePlans.size() != 1)
	    return std::nullopt;
	pre.choice = prePlans.front();
	int kp = pre.leaves.size();
	// a single solve leaf, or a solve from the right with an LU
	// factorisation, gives no operator
	if (pre.choice[0][kp - 1].kind == Choice::None)
	    return std::nullopt;

	Location loc = mul.getLoc();
	OpBuilder::InsertionGuard guard(b);
	b.setInsertionPoint(loop);
	auto cst = [&](int64_t v) {
	    return arith::ConstantIndexOp::create(b, loc, v).getResult();
	};
	auto mulI = [&](Value x, Value y) {
	    return arith::MulIOp::create(b, loc, x, y).getResult();
	};
	auto addI = [&](Value x, Value y) {
	    return arith::AddIOp::create(b, loc, x, y).getResult();
	};
	Value span = arith::SubIOp::create(b, loc, loop.getUpperBound(),
	                                   loop.getLowerBound());
	Value q = arith::CeilDivSIOp::create(b, loc, span, loop.getStep())
	              .getResult();
	// lb < ub ? q : 0 (a max with 0 would become arith.maxsi, which has no
	// EmitC lowering)
	Value trips = arith::SelectOp::create(
	    b, loc,
	    arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
	                          loop.getLowerBound(), loop.getUpperBound()),
	    q, cst(0));
	Value costP = costValue(pre, 0, kp - 1, loc);
	// Per step only matrix-vector kernels run if op(y) is a vector
	// (trsv, gemv): memory bound, they reach about a tenth of the flop
	// rate of the matrix-matrix kernels that form P (trsm, gemm), on the
	// Apple M5 as on Zen 4.  So a flop of the step counts kLevel2 times.
	int64_t w = isVector(last.cols()) ? kLevel2 : 1;
	Value apply = mulI(
	    cst(2 * w), mulI(mulI(dimIndex(pre.leaves[0].rowsDim(), loc),
	                          dimIndex(pre.leaves[kp - 1].colsDim(), loc)),
	                     dimIndex(last.colsDim(), loc)));
	c.choice = plan;
	Value costStep = mulI(cst(w), costValue(c, 0, k - 1, loc));
	Value cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
	                                   addI(costP, mulI(trips, apply)),
	                                   mulI(trips, costStep));

	// before the loop: P, if the condition holds
	Type elt = elementType(last.v);
	bool saved = inBranch;
	inBranch = true;
	Value e = empty(pre.leaves[0], pre.leaves[kp - 1], elt, loc);
	auto pIf = scf::IfOp::create(b, loc, TypeRange{e.getType()}, cond,
	                             /*withElse=*/true);
	{
	    OpBuilder::InsertionGuard g(b);
	    b.setInsertionPointToStart(pIf.thenBlock());
	    // computed in e itself, so that both branches yield e's buffer
	    Value pv = emitInto(pre, 0, kp - 1, e, 1.0, loc);
	    if (pv.getType() != e.getType())
		pv = AssumeOp::create(b, loc, e.getType(), pv, false);
	    scf::YieldOp::create(b, loc, pv);
	    b.setInsertionPointToStart(pIf.elseBlock());
	    scf::YieldOp::create(b, loc, e);
	}
	Leaf op;
	op.v = pIf.getResult(0);
	op.trans = transposed;

	// in the loop: P op(y), or the chain as before
	b.setInsertionPoint(mul);
	Type resultTy = mul.getType();
	auto fit = [&](Value r) {
	    return r.getType() == resultTy
	               ? r
		       : AssumeOp::create(b, loc, resultTy, r, false)
		             .getResult();
	};
	// both variants computed into the same destination d (one buffer,
	// hoisted out of the loop by bufferization)
	Value d = *newDest(mul);
	auto ifOp = scf::IfOp::create(b, loc, TypeRange{resultTy}, cond,
	                              /*withElse=*/true);
	{
	    OpBuilder::InsertionGuard g(b);
	    b.setInsertionPointToStart(ifOp.thenBlock());
	    scf::YieldOp::create(
	        b, loc, fit(gemm(op, last, d, alpha * c.alpha, 0.0, loc)));
	    b.setInsertionPointToStart(ifOp.elseBlock());
	    c.choice = plan;
	    scf::YieldOp::create(
	        b, loc, fit(emitInto(c, 0, k - 1, d, alpha * c.alpha, loc)));
	}
	inBranch = saved;
	return ifOp.getResult(0);
    }

    FailureOr<Value>
    selectProduct(MulOp mul)
    {
	return chain(mul, {}, 1.0, 0.0);
    }

    FailureOr<Value>
    selectAdd(AddOp add)
    {
	// products with their constant scale and the scalars known at run
	// time that scale them (s (A B) in a sum: alpha of the gemm)
	struct Product
	{
	    MulOp mul;
	    double alpha;
	    SmallVector<std::pair<Value, bool>> scales;
	};
	SmallVector<Product> products;
	SmallVector<std::pair<Value, double>> plain;
	// alpha * I: added to the diagonal at the end (adddiag), alpha the
	// product of the scales, constants or values known at run time
	SmallVector<SmallVector<Value>> identities;
	for (Value t : add.getTerms()) {
	    SmallVector<Value> alphas;
	    Value v = t;
	    while (auto sc = v.getDefiningOp<ScaleOp>()) {
		alphas.push_back(sc.getAlpha());
		v = sc.getOperand();
	    }
	    while (auto as = v.getDefiningOp<AssumeOp>())
		v = as.getOperand();
	    if (v.getDefiningOp<IdentityOp>()) {
		identities.push_back(alphas);
		continue;
	    }
	    if (auto pr = scaledProductAs<Product>(t)) {
		products.push_back(*pr);
		continue;
	    }
	    auto [inner, alpha] = peelScale(t);
	    auto mul = inner.getDefiningOp<MulOp>();
	    if (mul && inner.hasOneUse() && (inner == t || t.hasOneUse()))
		products.push_back({mul, alpha, {}});
	    else
		plain.push_back({inner, alpha});
	}
	// Mostly transposed terms (a Julia matrix is the transpose of the
	// row-major value): the sum is computed transposed, S^T = sum of the
	// transposed terms, so that they need no transposed copy; its users
	// (products) read S^T transposed (a gemm flag).  R = X W + 1 b^T - Y
	// becomes R^T = W^T X^T + b 1^T - Y^T, as an expert computes it with
	// column-major Y.
	bool flip = false;
	{
	    int tr = 0, nt = 0;
	    for (auto &[v, alpha] : plain)
		++(v.getDefiningOp<TransposeOp>() ? tr : nt);
	    auto at = cast<RankedTensorType>(add.getType());
	    bool sym =
	        bitEnumContainsAll(closure(propertiesOf(at), at.getShape()),
		                   Properties::symmetric);
	    // (a square or dynamic shape: the transposed products have the
	    // types of the products)
	    bool square = at.getDimSize(0) == at.getDimSize(1);
	    // (products, or transposes read by products: R^T ones)
	    bool users = llvm::all_of(add->getUsers(), [](Operation *u) {
		return isa<MulOp>(u) ||
		       (isa<TransposeOp>(u) &&
		        llvm::all_of(u->getUsers(), [](Operation *w) {
		            return isa<MulOp>(w);
		        }));
	    });
	    flip = tr > nt && !products.empty() && identities.empty() && !sym &&
	           square && users && !add->getUsers().empty() && !inBranch;
	}
	// (a Gram product Z^T Z, also scaled: syrk updates the lower triangle)
	auto gram = [](MulOp m) {
	    auto fs = m.getFactors();
	    if (fs.size() != 2)
		return false;
	    auto t0 = fs[0].getDefiningOp<TransposeOp>();
	    auto t1 = fs[1].getDefiningOp<TransposeOp>();
	    return (t0 && t0.getOperand() == fs[1]) ||
	           (t1 && t1.getOperand() == fs[0]);
	};
	bool keepLower =
	    !flip && plain.size() == 1 &&
	    (lowerOnly.contains(plain[0].first) ||
	     storageOf(plain[0].first) == 1) &&
	    (!identities.empty() || !products.empty()) &&
	    llvm::all_of(products, [&](Product &p) { return gram(p.mul); });
	for (auto &[v, alpha] : plain) {
	    if (flip) {
		if (auto t = v.getDefiningOp<TransposeOp>()) {
		    v = full(t.getOperand()); // S^T needs Y^T: Y itself
		    continue;
		}
		Leaf l;
		l.v = full(v);
		l.trans = true;
		v = untransposed(l, add.getLoc()).v;
		continue;
	    }
	    // a transposed term is copied transposed (O(mn)); an inverse in a
	    // sum was formed before (formInverses)
	    if (auto t = v.getDefiningOp<TransposeOp>()) {
		Leaf l;
		l.v = full(t.getOperand());
		l.trans = true;
		v = untransposed(l, add.getLoc()).v;
		continue;
	    }
	    // geaxpy / gescal read the whole matrix; a single matrix stored as
	    // its lower triangle plus identities (A + gamma I) stays so:
	    // adddiag changes only the diagonal
	    if (keepLower)
		continue;
	    v = full(v);
	}
	Location loc = add.getLoc();
	Value acc;
	double beta = 0.0;
	if (!plain.empty()) {
	    acc = plain[0].first;
	    beta = plain[0].second;
	    // with other plain terms (or no products) apply the scale now
	    if (beta != 1 && (plain.size() > 1 || products.empty())) {
		acc = lk::GescalOp::create(
		          b, loc, TypeRange{acc.getType()},
		          constant(elementType(acc), beta, loc), acc)
		          ->getResult(0);
		beta = 1;
	    }
	    for (auto &[v, alpha] : ArrayRef(plain).drop_front()) {
		acc = lk::GeaxpyOp::create(
		          b, loc, TypeRange{acc.getType()},
		          constant(elementType(acc), alpha, loc), v, acc)
		          ->getResult(0);
		beta = 1;
	    }
	}
	bool savedSandwich = noSandwich;
	if (flip) {
	    transposedProduct = true;
	    noSandwich = true;
	}
	for (Product &pr : products) {
	    MulOp mul = pr.mul;
	    // (a one-triangle accumulator: only syrk updates it as such; a
	    // gemm reads and writes the whole matrix)
	    if (acc && lowerOnly.contains(acc) && !gram(mul) &&
	        pr.scales.empty())
		acc = completeNow(acc, loc);
	    FailureOr<Value> r = chain(mul, acc, pr.alpha, beta, pr.scales);
	    if (failed(r)) {
		transposedProduct = false;
		noSandwich = savedSandwich;
		return failure();
	    }
	    acc = *r;
	    beta = 1.0;
	    consumed.insert(mul);
	}
	transposedProduct = false;
	noSandwich = savedSandwich;
	// (a product without accumulator is emitted at its own position; the
	// scalars of the identity terms may be defined after it)
	b.setInsertionPoint(add);
	if (!identities.empty() && !acc)
	    return add.emitError("ulm-lam-select-kernels: a sum of identities "
	                         "only (not supported yet)"),
	           failure();
	for (ArrayRef<Value> alphas : identities) {
	    Type elt = elementType(acc);
	    Value a;
	    for (Value x : alphas)
		a = a ? arith::MulFOp::create(b, loc, a, x).getResult() : x;
	    if (!a)
		a = constant(elt, 1.0, loc);
	    bool lower = lowerOnly.contains(acc) || keepLower;
	    acc =
	        lk::AdddiagOp::create(b, loc, TypeRange{acc.getType()}, a, acc)
	            ->getResult(0);
	    if (lower) // the diagonal belongs to the stored triangle too
		lowerOnly.insert(acc);
	}
	if (flip) // S = (S^T)^T: its users read it transposed
	    return TransposeOp::create(b, loc, acc).getResult();
	return acc;
    }

    /// Replace a ulm_lam result by a kernel result.  Different properties
    /// in the types are bridged with ulm_lam.assume (removed by
    /// ulm-lam-strip-properties).
    void
    replace(Operation *op, Value v)
    {
	Type t = op->getResult(0).getType();
	if (lowerOnly.contains(v)) // only the lower triangle is computed
	    t = withProperties(cast<RankedTensorType>(t),
	                       propertiesOf(t) | Properties::symmetric |
	                           Properties::lower_stored);
	// (always an assume for one triangle: storageOf ignores the storage
	// in the type of a kernel result, e.g. syrk on a lower-stored C)
	if (v.getType() != t || lowerOnly.contains(v)) {
	    b.setInsertionPoint(op);
	    v = AssumeOp::create(b, op->getLoc(), t, v, false);
	}
	op->getResult(0).replaceAllUsesWith(v);
	keepStorage(v);
    }

    /// An ulm_lam.assume of a value stored as one triangle (lam.assume(X^T
    /// X + s I, "spd") in a program) only adds facts about the value; it
    /// keeps the storage, so that its users (potrf, symm) read the stored
    /// triangle instead of a completed copy.
    void
    keepStorage(Value v)
    {
	int st = storageOf(v);
	if (!st)
	    return;
	for (Operation *u : v.getUsers()) {
	    auto as = dyn_cast<AssumeOp>(u);
	    // only an assume that asserts properties (lam.assume): one to the
	    // plain type is how the frontends hand a value to return, yield or
	    // a call, whose types must not change
	    if (!as || storageOf(as.getResult()) ||
	        propertiesOf(as.getType()) == Properties::general ||
	        llvm::any_of(as->getUsers(), [](Operation *w) {
		    return w->hasTrait<OpTrait::IsTerminator>() ||
		           isa<func::CallOp>(w);
	        }))
		continue;
	    auto t = cast<RankedTensorType>(as.getType());
	    as.getResult().setType(
	        withProperties(t, propertiesOf(t) | Properties::symmetric |
		                      (st == 1 ? Properties::lower_stored
		                               : Properties::upper_stored)));
	    keepStorage(as.getResult());
	}
    }

    LogicalResult
    cleanup()
    {
	// Erase ulm_lam ops that are dead now, users before producers.
	bool changed = true;
	while (changed) {
	    changed = false;
	    SmallVector<Operation *> dead;
	    func.walk([&](Operation *op) {
		if (isUlmLamOp(op) && !isa<AssumeOp>(op) && op->use_empty())
		    dead.push_back(op);
	    });
	    for (Operation *op : dead) {
		op->erase();
		changed = true;
	    }
	}
	// A transpose that no kernel absorbed (a flag of gemm, trsm, ...) is
	// a transposed copy.
	SmallVector<TransposeOp> left;
	func.walk([&](TransposeOp t) { left.push_back(t); });
	for (TransposeOp t : left) {
	    b.setInsertionPoint(t);
	    Leaf l;
	    l.v = full(t.getOperand());
	    l.trans = true;
	    replace(t, untransposed(l, t.getLoc()).v);
	    t->erase();
	}
	WalkResult res = func.walk([&](Operation *op) {
	    if (isa<InverseOp>(op)) {
		op->emitError(
		    "ulm-lam-select-kernels: explicit inverse (not a factor "
		    "of a product); solve with it instead");
		return WalkResult::interrupt();
	    }
	    if (isUlmLamOp(op) && !isa<AssumeOp>(op)) {
		op->emitError(
		    "ulm-lam-select-kernels: no kernel sequence found for "
		    "this op");
		return WalkResult::interrupt();
	    }
	    return WalkResult::advance();
	});
	return failure(res.wasInterrupted());
    }
};

struct UlmLamSelectKernels
    : PassWrapper<UlmLamSelectKernels, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamSelectKernels)
    UlmLamSelectKernels() = default;
    UlmLamSelectKernels(const UlmLamSelectKernels &other) : PassWrapper(other)
    {
    }
    StringRef
    getArgument() const final
    {
	return "ulm-lam-select-kernels";
    }
    StringRef
    getDescription() const final
    {
	return "Map ulm_lam expressions onto ulm_lamk kernels (solve the LAMP)";
    }
    void
    getDependentDialects(DialectRegistry &registry) const override
    {
	registry.insert<lk::UlmLamkDialect, tensor::TensorDialect,
	                bufferization::BufferizationDialect,
	                arith::ArithDialect, scf::SCFDialect>();
    }
    Option<std::string> strategy{*this, "strategy",
                                 llvm::cl::desc("gmc | search | egraph"),
                                 llvm::cl::init("gmc")};
    Option<bool> lapack{
        *this, "lapack",
        llvm::cl::desc("the backend has LAPACK (CBLAS/LAPACK): L D L^T "
	               "(sytrf) for symmetric matrices not known to be spd, "
	               "least squares by QR (gels)"),
        llvm::cl::init(false)};
    Option<bool> expandFactors{
        *this, "expand-factors",
        llvm::cl::desc("a factor that is a computed product may be replaced "
	               "by its factors where that is cheaper (K H P with "
	               "K = P H^T S^-1: syrk)"),
        llvm::cl::init(true)};
    Option<bool> distributeSums{
        *this, "distribute",
        llvm::cl::desc("a product with a sum as a factor may be distributed "
	               "where that is cheaper ((W + B A) x = W x + B (A x))"),
        llvm::cl::init(true)};
    void
    runOnOperation() override
    {
	if (strategy != "gmc") {
	    getOperation().emitError(
	        "ulm-lam-select-kernels: only strategy=gmc is "
	        "implemented");
	    return signalPassFailure();
	}
	Selector sel(getOperation(), lapack, expandFactors);
	sel.distributeSums = distributeSums;
	if (failed(sel.run()))
	    signalPassFailure();
    }
};

//===----------------------------------------------------------------------===//
// ulm-lam-strip-properties
//===----------------------------------------------------------------------===//

struct UlmLamStripProperties
    : PassWrapper<UlmLamStripProperties, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamStripProperties)
    StringRef
    getArgument() const final
    {
	return "ulm-lam-strip-properties";
    }
    StringRef
    getDescription() const final
    {
	return "Drop #ulm_lam.props encodings (and ulm_lam.assume) before "
	       "bufferization";
    }
    void
    runOnOperation() override
    {
	getOperation().walk([](Operation *op) {
	    for (Value r : op->getResults())
		r.setType(stripProperties(r.getType()));
	    for (Region &region : op->getRegions())
		for (Block &block : region)
		    for (BlockArgument a : block.getArguments())
			a.setType(stripProperties(a.getType()));
	    if (auto f = dyn_cast<FunctionOpInterface>(op)) {
		auto ft = cast<FunctionType>(f.getFunctionType());
		// the storage of a parameter stays visible (interface comment)
		for (auto [i, t] : llvm::enumerate(ft.getInputs())) {
		    Properties p = propertiesOf(t);
		    if (bitEnumContainsAll(p, Properties::lower_stored))
			f.setArgAttr(
			    i, "ulm_lam.stored",
			    StringAttr::get(op->getContext(), "lower"));
		    if (bitEnumContainsAll(p, Properties::upper_stored))
			f.setArgAttr(
			    i, "ulm_lam.stored",
			    StringAttr::get(op->getContext(), "upper"));
		}
		auto strip = [](TypeRange ts) {
		    return llvm::map_to_vector(
		        ts, [](Type t) { return stripProperties(t); });
		};
		f.setType(FunctionType::get(op->getContext(),
		                            strip(ft.getInputs()),
		                            strip(ft.getResults())));
	    }
	});
	// Without properties every ulm_lam.assume is the identity.
	SmallVector<AssumeOp> assumes;
	getOperation().walk([&](AssumeOp a) { assumes.push_back(a); });
	for (AssumeOp a : assumes) {
	    a.getResult().replaceAllUsesWith(a.getOperand());
	    a.erase();
	}
    }
};

} // namespace

namespace ulm_lam {
void
registerUlmLamPasses()
{
    PassRegistration<UlmLamNormalize>();
    PassRegistration<UlmLamInferProperties>();
    PassRegistration<UlmLamSelectKernels>();
    PassRegistration<UlmLamStripProperties>();
}
} // namespace ulm_lam
