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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>
#include <map>

using namespace mlir;
using namespace mlir::ulm_lam;
namespace lk = mlir::ulm_lamk;

namespace {

bool
isUlmLamOp(Operation *op)
{
    return isa_and_nonnull<UlmLamDialect>(op->getDialect());
}

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

    RankedTensorType
    type() const
    {
	return cast<RankedTensorType>(v.getType());
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
};

class Selector
{
  public:
    explicit Selector(func::FuncOp f) : func(f), b(f.getContext()), dom(f) {}

    LogicalResult
    run()
    {
	SmallVector<Operation *> ops;
	func.walk([&](Operation *op) {
	    if (isa<MulOp, AddOp>(op))
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
	    FailureOr<Value> r = isa<AddOp>(op)
	                             ? selectAdd(cast<AddOp>(op))
	                             : selectProduct(cast<MulOp>(op));
	    if (failed(r))
		return failure();
	    replace(op, *r);
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

    static Key
    keyOf(ArrayRef<Leaf> leaves, int i, int j, bool transposed)
    {
	Key k;
	for (int t = 0; t <= j - i; ++t) {
	    const Leaf &l = transposed ? leaves[j - t] : leaves[i + t];
	    k.push_back(l.kind);
	    k.push_back(reinterpret_cast<intptr_t>(l.v.getAsOpaquePointer()));
	    k.push_back(reinterpret_cast<intptr_t>(l.piv.getAsOpaquePointer()));
	    k.push_back(l.trans != transposed);
	    k.push_back(l.lower);
	}
	return k;
    }

    static bool
    feedsSum(Operation *op)
    {
	if (!isa<MulOp>(op) || !op->hasOneUse())
	    return false;
	Operation *user = *op->getUsers().begin();
	if (isa<ScaleOp>(user) && user->hasOneUse())
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
	if (bitEnumContainsAll(p, Properties::lower_tri) ||
	    bitEnumContainsAll(p, Properties::upper_tri)) {
	    f = {Factor::Triangular,
	         m,
	         {},
	         bitEnumContainsAll(p, Properties::lower_tri)};
	} else {
	    OpBuilder::InsertionGuard guard(b);
	    if (Operation *def = m.getDefiningOp())
		b.setInsertionPointAfter(def);
	    else
		b.setInsertionPointToStart(m.getParentBlock());
	    Location loc = m.getLoc();
	    if (bitEnumContainsAll(p, Properties::spd)) {
		auto potrf = lk::PotrfOp::create(b, loc, TypeRange{m.getType()},
		                                 lk::Uplo::lower, m,
		                                 /*checked=*/true);
		f = {Factor::Cholesky, potrf->getResult(0), {}, true};
	    } else {
		auto pivTy = RankedTensorType::get(
		    {t.getDimSize(0)}, IntegerType::get(b.getContext(), 32));
		SmallVector<Value> n;
		if (pivTy.isDynamicDim(0))
		    n.push_back(dimValue(b, loc, m, 0));
		Value piv = tensor::EmptyOp::create(b, loc, pivTy, n);
		auto getrf =
		    lk::GetrfOp::create(b, loc, TypeRange{m.getType(), pivTy},
		                        m, piv, /*checked=*/true);
		f = {Factor::LU, getrf->getResult(0), getrf->getResult(1),
		     true};
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
		if (!matchPattern(sc.getAlpha(), m_Constant(&c)))
		    return user->emitError(
		        "ulm-lam-select-kernels: a factor scaled by a "
		        "non-constant scalar (not supported yet)");
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
	    out.push_back({Leaf::Mat, v, {}, trans, true});
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
	case Factor::Cholesky: // (L L^T)^-1 = L^-T L^-1, also for the transpose
	    out.push_back({Leaf::TriSolve, f.f, {}, true, true});
	    out.push_back({Leaf::TriSolve, f.f, {}, false, true});
	    break;
	case Factor::LU:
	    out.push_back({Leaf::LUSolve, f.f, f.piv, trans, true});
	    break;
	}
	return success();
    }

    // ---- the generalised matrix chain --------------------------------------

    struct Chain
    {
	SmallVector<Leaf> leaves;
	double alpha = 1.0; // constant factors moved out of the product
	std::vector<std::vector<double>> cost;
	std::vector<std::vector<Choice>> choice;
    };

    static bool
    isVector(int64_t cols)
    {
	return cols == 1;
    }

    /// Interval [i, j] can be the in-place operand of a solve: it is computed
    /// into a buffer, or it is a single untransposed matrix.
    static bool
    solveOperand(const Chain &c, int i, int j)
    {
	if (i < j)
	    return c.cost[i][j] < kInf &&
	           !(c.choice[i][j].kind == Choice::Reuse &&
	             c.choice[i][j].trans);
	return c.leaves[i].kind == Leaf::Mat && !c.leaves[i].trans;
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
		double m = costSize(L[i].rows()), n = costSize(L[j].cols());
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
		    double inner = costSize(L[s].cols());
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
			consider(i, j, c.cost[s + 1][j] + f * m * m * n,
			         {Choice::SolveLeft, s});
		    }
		    // a triangular solve applied from the right to the left
		    // part
		    if (s + 1 == j && L[j].kind == Leaf::TriSolve &&
		        solveOperand(c, i, s))
			consider(i, j, c.cost[i][s] + m * n * n,
			         {Choice::SolveRight, s});
		}
		// Z^T Z: the left half is the transpose of the right half
		if (len % 2 == 0) {
		    int h = i + len / 2;
		    bool mirror = true;
		    for (int t = 0; t < len / 2 && mirror; ++t)
			mirror = L[i + t].isTransposeOf(L[j - t]);
		    if (mirror && c.cost[h][j] < kInf) {
			double zr = costSize(L[h].rows());
			double cost = c.cost[h][j] + 2 * n * zr * n;
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
	if (s.kind == Leaf::LUSolve)
	    return lk::GetrsOp::create(b, loc, TypeRange{rhs.getType()}, trans,
	                               s.v, s.piv, rhs)
	        ->getResult(0);
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
	if (beta != 1)
	    dest = lk::GescalOp::create(b, loc, TypeRange{dest.getType()},
	                                constant(elt, beta, loc), dest)
	               ->getResult(0);
	return lk::GeaxpyOp::create(b, loc, TypeRange{dest.getType()},
	                            constant(elt, alpha, loc), x, dest)
	    ->getResult(0);
    }

    Value
    gemm(Leaf l, Leaf r, Value dest, double alpha, double beta, Location loc)
    {
	Type elt = elementType(l.v);
	if (!dest)
	    dest = empty(l, r, elt, loc);
	auto t = [](bool x) { return x ? lk::Trans::trans : lk::Trans::none; };
	if (isVector(r.cols()) && !r.trans)
	    return lk::GemvOp::create(b, loc, TypeRange{dest.getType()},
	                              t(l.trans), constant(elt, alpha, loc),
	                              l.v, r.v, constant(elt, beta, loc), dest)
	        ->getResult(0);
	return lk::GemmOp::create(b, loc, TypeRange{dest.getType()}, t(l.trans),
	                          t(r.trans), constant(elt, alpha, loc), l.v,
	                          r.v, constant(elt, beta, loc), dest)
	    ->getResult(0);
    }

    /// Emit the product of leaves [i, j] as chosen by the DP; the result is
    /// beta * dest + alpha * product (dest may be null: beta = 0).
    Value
    emit(Chain &c, int i, int j, Value dest, double alpha, double beta,
         Location loc)
    {
	Value r = emitKernels(c, i, j, dest, alpha, beta, loc);
	// a plain product: remember it for later products
	if (alpha == 1 && beta == 0 &&
	    (!dest || dest.getDefiningOp<tensor::EmptyOp>()))
	    computed[keyOf(c.leaves, i, j, false)] = r;
	return r;
    }

    Value
    emitKernels(Chain &c, int i, int j, Value dest, double alpha, double beta,
                Location loc)
    {
	Choice ch = c.choice[i][j];
	int s = ch.split;
	switch (ch.kind) {
	case Choice::Gemm:
	    return gemm(operand(c, i, s, loc), operand(c, s + 1, j, loc), dest,
	                alpha, beta, loc);
	case Choice::Gram: {
	    Leaf z = operand(c, s, j, loc);
	    Leaf zt = z;
	    zt.trans = !z.trans;
	    return gemm(zt, z, dest, alpha, beta, loc);
	}
	case Choice::SolveLeft: {
	    Leaf rhs = operand(c, s + 1, j, loc);
	    return accumulate(solveInPlace(c.leaves[i], rhs.v, loc), dest,
	                      alpha, beta, loc);
	}
	case Choice::SolveRight: {
	    Leaf lhs = operand(c, i, s, loc);
	    const Leaf &t = c.leaves[j];
	    Value x = lk::TrsmOp::create(
	                  b, loc, TypeRange{lhs.v.getType()}, lk::Side::right,
	                  t.lower ? lk::Uplo::lower : lk::Uplo::upper,
	                  t.trans ? lk::Trans::trans : lk::Trans::none,
	                  lk::Diag::non_unit,
	                  constant(elementType(lhs.v), 1, loc), t.v, lhs.v)
	                  ->getResult(0);
	    return accumulate(x, dest, alpha, beta, loc);
	}
	case Choice::Reuse: // only proper subintervals, used via operand()
	case Choice::None:
	    break;
	}
	llvm_unreachable("interval without a kernel sequence");
    }

    /// beta * dest + alpha * (product), dest null: a new value.
    FailureOr<Value>
    chain(MulOp mul, Value dest, double alpha, double beta)
    {
	Chain c;
	for (Value f : mul.getFactors())
	    if (failed(leavesOf(f, mul, c.leaves, c.alpha)))
		return failure();
	solveChain(c, mul);
	int k = c.leaves.size();
	if (c.cost[0][k - 1] == kInf)
	    return mul.emitError(
	               "ulm-lam-select-kernels: no kernel sequence for this "
	               "product (an explicit inverse, or a right solve "
	               "with an LU factorisation, would be needed)"),
	           failure();
	b.setInsertionPoint(mul);
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

    FailureOr<Value>
    selectProduct(MulOp mul)
    {
	return chain(mul, {}, 1.0, 0.0);
    }

    FailureOr<Value>
    selectAdd(AddOp add)
    {
	SmallVector<std::pair<MulOp, double>> products;
	SmallVector<std::pair<Value, double>> plain;
	for (Value t : add.getTerms()) {
	    auto [inner, alpha] = peelScale(t);
	    auto mul = inner.getDefiningOp<MulOp>();
	    if (mul && inner.hasOneUse() && (inner == t || t.hasOneUse()))
		products.push_back({mul, alpha});
	    else
		plain.push_back({inner, alpha});
	}
	for (auto &[v, alpha] : plain)
	    if (v.getDefiningOp<TransposeOp>() || v.getDefiningOp<InverseOp>())
		return add.emitError(
		           "ulm-lam-select-kernels: transposed or inverted "
		           "term of a sum (not supported yet)"),
		       failure();
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
	for (auto [mul, alpha] : products) {
	    FailureOr<Value> r = chain(mul, acc, alpha, beta);
	    if (failed(r))
		return failure();
	    acc = *r;
	    beta = 1.0;
	    consumed.insert(mul);
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
	if (v.getType() != t) {
	    b.setInsertionPoint(op);
	    v = AssumeOp::create(b, op->getLoc(), t, v, false);
	}
	op->getResult(0).replaceAllUsesWith(v);
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
	                arith::ArithDialect>();
    }
    Option<std::string> strategy{*this, "strategy",
                                 llvm::cl::desc("gmc | search | egraph"),
                                 llvm::cl::init("gmc")};
    void
    runOnOperation() override
    {
	if (strategy != "gmc") {
	    getOperation().emitError(
	        "ulm-lam-select-kernels: only strategy=gmc is "
	        "implemented");
	    return signalPassFailure();
	}
	if (failed(Selector(getOperation()).run()))
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
