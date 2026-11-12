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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
	patterns.add<ForwardDroppingAssume>(ctx);
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
	return kind == o.kind && v == o.v && piv == o.piv && lower == o.lower &&
	       trans != o.trans;
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
    explicit Selector(func::FuncOp f, bool lapack = false)
        : func(f), b(f.getContext()), dom(f), lapack(lapack)
    {
    }
    /// The backend has LAPACK (CBLAS/LAPACK): L D L^T (sytrf) for symmetric
    /// matrices that are not known to be spd, instead of LU; least squares
    /// by QR (gels) instead of the normal equations.
    bool lapack = false;

    LogicalResult
    run()
    {
	if (!lapack)
	    normalEquations();
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
	    replace(op, *r);
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
		if (r.getType() != v.getType())
		    r = AssumeOp::create(b, yield.getLoc(), v.getType(), r,
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
	for (OpOperand *o : uses)
	    o->set(full(o->get()));
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
	Value r = lk::GescalOp::create(b, loc, TypeRange{x.getType()}, alpha, x)
	              ->getResult(0);
	if (lowerOnly.contains(x))
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
	Factor f = factorise(v);
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
	return {Leaf::Mat, emit(c, i, j, {}, 1, 0, loc), {}, false, true};
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
	if (beta != 1)
	    dest = lk::GescalOp::create(b, loc, TypeRange{dest.getType()},
	                                constant(elt, beta, loc), dest)
	               ->getResult(0);
	return lk::GeaxpyOp::create(b, loc, TypeRange{dest.getType()},
	                            constant(elt, alpha, loc), x, dest)
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
	    bool top = i == 0 && j + 1 == static_cast<int>(c.leaves.size());
	    bool fresh =
	        beta == 0 && (!dest || dest.getDefiningOp<tensor::EmptyOp>());
	    bool update = beta != 0 && dest && symmetricAcc(dest);
	    if (top && !inBranch && z.stored == Leaf::Full &&
	        (fresh || update)) {
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
	for (size_t i = 0; i < a.size(); ++i)
	    for (size_t j = 0; j < a.size(); ++j) {
		const Choice &x = a[i][j], &y = b[i][j];
		if (x.kind != y.kind || x.split != y.split ||
		    x.value != y.value || x.trans != y.trans)
		    return false;
	    }
	return true;
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
	for (Value f : mul.getFactors())
	    if (failed(leavesOf(f, mul, c.leaves, c.alpha)))
		return failure();
	dynScales.append(extraScales.begin(), extraScales.end());
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
	    // (a scalar defined after the product, e.g. 1/a in A^-1 x / a:
	    // the scaling goes after it)
	    for (auto &sc : scales) {
		Operation *d = sc.first.getDefiningOp();
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
	    return accumulate(x, dest, 1.0, beta, mul.getLoc());
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
	for (auto &[v, alpha] : plain) {
	    // a transposed term is copied transposed (O(mn)); an inverse in a
	    // sum was formed before (formInverses)
	    if (auto t = v.getDefiningOp<TransposeOp>()) {
		Leaf l;
		l.v = full(t.getOperand());
		l.trans = true;
		v = untransposed(l, add.getLoc()).v;
		continue;
	    }
	    v = full(v); // geaxpy / gescal read the whole matrix
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
	for (Product &pr : products) {
	    MulOp mul = pr.mul;
	    FailureOr<Value> r = chain(mul, acc, pr.alpha, beta, pr.scales);
	    if (failed(r))
		return failure();
	    acc = *r;
	    beta = 1.0;
	    consumed.insert(mul);
	}
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
	    bool lower = lowerOnly.contains(acc);
	    acc =
	        lk::AdddiagOp::create(b, loc, TypeRange{acc.getType()}, a, acc)
	            ->getResult(0);
	    if (lower) // the diagonal belongs to the stored triangle too
		lowerOnly.insert(acc);
	}
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
    void
    runOnOperation() override
    {
	if (strategy != "gmc") {
	    getOperation().emitError(
	        "ulm-lam-select-kernels: only strategy=gmc is "
	        "implemented");
	    return signalPassFailure();
	}
	if (failed(Selector(getOperation(), lapack).run()))
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
