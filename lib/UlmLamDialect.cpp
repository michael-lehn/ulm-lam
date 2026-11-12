//===- UlmLamDialect.cpp - ulm_lam and ulm_lamk dialects ------------------===//
//
// Property algebra, type inference, verifiers, canonicalisation patterns,
// memory effects, costs and bufferization of the ulm_lam and ulm_lamk ops.
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ulm_lam;

#include "UlmLamEnums.cpp.inc"

#include "UlmLamDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "UlmLamAttrs.cpp.inc"

#include "UlmLamInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "UlmLamOps.cpp.inc"
#define GET_OP_CLASSES
#include "UlmLamkOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialects
//===----------------------------------------------------------------------===//

void
UlmLamDialect::initialize()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "UlmLamAttrs.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "UlmLamOps.cpp.inc"
        >();
}

void
UlmLamDialect::getCanonicalizationPatterns(RewritePatternSet &) const
{
}

// ulm_lamk::UlmLamkDialect::initialize() is in UlmLamkDialect.cpp (separate
// translation unit: the generated attribute parsers of both dialects
// would collide).

//===----------------------------------------------------------------------===//
// Property algebra
//===----------------------------------------------------------------------===//

namespace {
using P = mlir::ulm_lam::Properties;

bool
has(P set, P bits)
{
    return bitEnumContainsAll(set, bits);
}

bool
isStaticSquare(ArrayRef<int64_t> shape)
{
    return shape.size() == 2 && !ShapedType::isDynamic(shape[0]) &&
           shape[0] == shape[1];
}
} // namespace

::mlir::ulm_lam::Properties
mlir::ulm_lam::propertiesOf(Type type)
{
    auto t = dyn_cast<RankedTensorType>(type);
    if (!t)
	return P::general;
    if (auto a = dyn_cast_or_null<PropertiesAttr>(t.getEncoding()))
	return a.getValue();
    return P::general;
}

::mlir::ulm_lam::Properties
mlir::ulm_lam::closure(::mlir::ulm_lam::Properties p, ArrayRef<int64_t> shape)
{
    P q = p;
    if (isStaticSquare(shape))
	q = q | P::square;
    for (int iter = 0; iter < 3; ++iter) { // implications chain at most 3 deep
	if (has(q, P::identity))
	    q = q | P::lower_tri | P::upper_tri | P::spd | P::orthogonal |
	        P::permutation | P::unit_diag;
	if (has(q, P::spd))
	    q = q | P::spsd | P::symmetric | P::non_singular | P::square;
	if (has(q, P::spsd))
	    q = q | P::symmetric | P::square;
	if (has(q, P::symmetric))
	    q = q | P::square;
	if (has(q, P::permutation))
	    q = q | P::orthogonal;
	if (has(q, P::orthogonal))
	    q = q | P::full_rank;
	if (has(q, P::non_singular))
	    q = q | P::full_rank | P::square;
	if (has(q, P::square | P::full_rank))
	    q = q | P::non_singular;
    }
    return q;
}

RankedTensorType
mlir::ulm_lam::withProperties(RankedTensorType type, Properties p)
{
    if (isStaticSquare(type.getShape()))
	p = p & ~P::square;
    Attribute enc;
    if (p != P::general)
	enc = PropertiesAttr::get(type.getContext(), p);
    return RankedTensorType::get(type.getShape(), type.getElementType(), enc);
}

Type
mlir::ulm_lam::stripProperties(Type type)
{
    auto t = dyn_cast<RankedTensorType>(type);
    if (!t || !isa_and_nonnull<PropertiesAttr>(t.getEncoding()))
	return type;
    return RankedTensorType::get(t.getShape(), t.getElementType());
}

LogicalResult
PropertiesAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type,
    function_ref<InFlightDiagnostic()> emitError) const
{
    P p = closure(getValue(), shape);
    if (shape.size() != 2)
	return emitError() << "#ulm_lam.props needs a rank-2 tensor";
    if (has(p, P::zero) && has(p, P::full_rank))
	return emitError() << "#ulm_lam.props: zero contradicts full rank";
    if (has(p, P::lower_stored | P::upper_stored))
	return emitError() << "#ulm_lam.props: lower_stored and upper_stored "
	                      "exclude each other";
    if ((has(p, P::lower_stored) || has(p, P::upper_stored)) &&
        !has(p, P::symmetric))
	return emitError() << "#ulm_lam.props: lower_stored / upper_stored "
	                      "need a symmetric matrix";
    if (has(p, P::square) && !ShapedType::isDynamic(shape[0]) &&
        !ShapedType::isDynamic(shape[1]) && shape[0] != shape[1])
	return emitError() << "#ulm_lam.props: square property on a "
	                   << shape[0] << "x" << shape[1] << " tensor";
    return success();
}

//===----------------------------------------------------------------------===//
// ulm_lam: type inference
//===----------------------------------------------------------------------===//

namespace {

int64_t
rows(Type t)
{
    return cast<RankedTensorType>(t).getDimSize(0);
}
int64_t
cols(Type t)
{
    return cast<RankedTensorType>(t).getDimSize(1);
}

bool
dimsCompatible(int64_t a, int64_t b)
{
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

/// The value a transpose leaf refers to, or null.
Value
transposed(Value v)
{
    if (auto t = v.getDefiningOp<TransposeOp>())
	return t.getOperand();
    return {};
}

bool
isSymmetric(Value v)
{
    return has(closure(propertiesOf(v.getType()),
                       cast<RankedTensorType>(v.getType()).getShape()),
               P::symmetric);
}

/// A factor of a product up to transposition: the matrix and whether it
/// is transposed (never for a symmetric matrix: S^T = S).
std::pair<Value, bool>
leafOf(Value v)
{
    bool t = false;
    while (Value a = transposed(v)) {
	v = a;
	t = !t;
    }
    return {v, isSymmetric(v) ? false : t};
}

/// Z has full column rank: a single factor of full rank with at least as
/// many rows as columns, or a product of non-singular factors.  With a
/// dynamic size, a full-rank Z is taken to have full column rank: Z^T M Z
/// is then invertible, as the formulas that invert it require (if not,
/// the checked Cholesky factorisation reports it at run time).
bool
fullColumnRank(ValueRange z, ArrayRef<P> ps)
{
    if (llvm::all_of(ps, [](P p) { return has(p, P::non_singular); }))
	return true;
    if (z.size() != 1 || !has(ps[0], P::full_rank))
	return false;
    int64_t r = rows(z[0].getType()), c = cols(z[0].getType());
    return ShapedType::isDynamic(r) || ShapedType::isDynamic(c) || r >= c;
}

/// Products that read the same backwards when transposed,
/// F_1 ... F_k with F_i^T = F_(k+1-i):
///   Z^T Z      (k even)            spsd, spd if Z has full column rank;
///   Z^T M Z    (k odd, M middle)   symmetric if M is, spsd if M is, spd if
///                                  M is spd and Z has full column rank.
/// This covers A^T A, A A^T, A^T P A, and S S, S^3 for symmetric S (S^T =
/// S): e.g. R R with R spd is spd.
P
palindromeProperties(ValueRange f, ArrayRef<P> ps)
{
    size_t k = f.size(), h = k / 2;
    if (k < 2)
	return P::general;
    for (size_t i = 0; i < h; ++i) {
	auto [a, ta] = leafOf(f[i]);
	auto [b, tb] = leafOf(f[k - 1 - i]);
	if (a != b || (ta == tb && !isSymmetric(a)))
	    return P::general;
    }
    ValueRange z = f.drop_front(k - h);
    ArrayRef<P> pz = ps.drop_front(k - h);
    if (k % 2 == 0)
	return fullColumnRank(z, pz) ? P::spd : P::spsd;
    P m = ps[h];
    if (has(m, P::spd) && fullColumnRank(z, pz))
	return P::spd;
    if (has(m, P::spsd))
	return P::spsd;
    if (has(m, P::symmetric))
	return P::symmetric;
    return P::general;
}

P
mulProperties(ValueRange f)
{
    SmallVector<P> ps;
    for (Value v : f)
	ps.push_back(closure(propertiesOf(v.getType()),
	                     cast<RankedTensorType>(v.getType()).getShape()));
    P all = P::lower_tri | P::upper_tri | P::unit_diag | P::orthogonal |
            P::permutation | P::non_singular | P::identity;
    P r = all;
    bool anyZero = false;
    for (P p : ps) {
	r = r & p;
	anyZero |= has(p, P::zero);
    }
    r = r & all;
    if (anyZero)
	return P::zero;
    // N A and A N with N non-singular keep the rank of A
    if (f.size() == 2 &&
        ((has(ps[0], P::non_singular) && has(ps[1], P::full_rank)) ||
         (has(ps[1], P::non_singular) && has(ps[0], P::full_rank))))
	r = r | P::full_rank;
    return r | palindromeProperties(f, ps);
    return r;
}

P
addProperties(ValueRange terms)
{
    P r = P::symmetric | P::spsd | P::lower_tri | P::upper_tri;
    bool anySpd = false;
    for (Value v : terms) {
	P p = closure(propertiesOf(v.getType()),
	              cast<RankedTensorType>(v.getType()).getShape());
	r = r & p;
	anySpd |= has(p, P::spd);
    }
    r = r & (P::symmetric | P::spsd | P::lower_tri | P::upper_tri);
    if (has(r, P::spsd) && anySpd)
	r = r | P::spd;
    return r;
}

P
transposeProperties(P p)
{
    // (storage is never propagated to the result of an op)
    P r =
        p & ~(P::lower_tri | P::upper_tri | P::lower_stored | P::upper_stored);
    if (has(p, P::lower_tri))
	r = r | P::upper_tri;
    if (has(p, P::upper_tri))
	r = r | P::lower_tri;
    return r;
}

P
inverseProperties(P p)
{
    return p & (P::lower_tri | P::upper_tri | P::unit_diag | P::symmetric |
                P::spd | P::orthogonal | P::permutation | P::identity |
                P::non_singular | P::full_rank | P::square);
}

P
scaleProperties(Value alpha, P p)
{
    P r = p & (P::lower_tri | P::upper_tri | P::symmetric | P::zero);
    FloatAttr c;
    if (matchPattern(alpha, m_Constant(&c))) {
	double a = c.getValueAsDouble();
	if (a > 0)
	    r = r | (p & (P::spd | P::spsd));
	if (a != 0)
	    r = r | (p & (P::non_singular | P::full_rank));
	else
	    r = P::zero;
    }
    return r;
}

RankedTensorType
tensorType(Type t)
{
    return cast<RankedTensorType>(t);
}

/// Shapes compatible, element types equal, properties of `actual` implied
/// by `inferred`.
bool
ulmLamCompatible(TypeRange inferred, TypeRange actual)
{
    if (inferred.size() != actual.size())
	return false;
    for (auto [i, a] : llvm::zip(inferred, actual)) {
	auto it = dyn_cast<RankedTensorType>(i),
	     at = dyn_cast<RankedTensorType>(a);
	if (!it || !at || it.getRank() != at.getRank() ||
	    it.getElementType() != at.getElementType())
	    return false;
	for (auto [x, y] : llvm::zip(it.getShape(), at.getShape()))
	    if (!dimsCompatible(x, y))
		return false;
	P have = closure(propertiesOf(it), it.getShape());
	if (!has(have, propertiesOf(at)))
	    return false;
    }
    return true;
}

} // namespace

#define ULM_LAM_COMPATIBLE(OP)                                                 \
    bool OP::isCompatibleReturnTypes(TypeRange inferred, TypeRange actual)     \
    {                                                                          \
	return ulmLamCompatible(inferred, actual);                             \
    }
ULM_LAM_COMPATIBLE(MulOp)
ULM_LAM_COMPATIBLE(AddOp)
ULM_LAM_COMPATIBLE(ScaleOp)
ULM_LAM_COMPATIBLE(TransposeOp)
ULM_LAM_COMPATIBLE(InverseOp)
#undef ULM_LAM_COMPATIBLE

LogicalResult
MulOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                        ValueRange operands, DictionaryAttr, PropertyRef,
                        RegionRange, SmallVectorImpl<Type> &inferred)
{
    if (operands.empty())
	return failure();
    auto first = tensorType(operands.front().getType());
    auto last = tensorType(operands.back().getType());
    auto t = RankedTensorType::get({first.getDimSize(0), last.getDimSize(1)},
                                   first.getElementType());
    inferred.push_back(withProperties(t, mulProperties(operands)));
    return success();
}

LogicalResult
AddOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                        ValueRange operands, DictionaryAttr, PropertyRef,
                        RegionRange, SmallVectorImpl<Type> &inferred)
{
    if (operands.empty())
	return failure();
    auto t = tensorType(operands.front().getType());
    // the most static shape among the terms
    SmallVector<int64_t> shape(t.getShape());
    for (Value v : operands)
	for (auto [i, d] : llvm::enumerate(tensorType(v.getType()).getShape()))
	    if (ShapedType::isDynamic(shape[i]))
		shape[i] = d;
    inferred.push_back(
        withProperties(RankedTensorType::get(shape, t.getElementType()),
	               addProperties(operands)));
    return success();
}

LogicalResult
ScaleOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                          ValueRange operands, DictionaryAttr, PropertyRef,
                          RegionRange, SmallVectorImpl<Type> &inferred)
{
    auto t = tensorType(operands[1].getType());
    P p = closure(propertiesOf(t), t.getShape());
    inferred.push_back(withProperties(t, scaleProperties(operands[0], p)));
    return success();
}

LogicalResult
TransposeOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                              ValueRange operands, DictionaryAttr, PropertyRef,
                              RegionRange, SmallVectorImpl<Type> &inferred)
{
    auto t = tensorType(operands[0].getType());
    auto r = RankedTensorType::get({t.getDimSize(1), t.getDimSize(0)},
                                   t.getElementType());
    inferred.push_back(withProperties(
        r, transposeProperties(closure(propertiesOf(t), t.getShape()))));
    return success();
}

LogicalResult
InverseOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                            ValueRange operands, DictionaryAttr, PropertyRef,
                            RegionRange, SmallVectorImpl<Type> &inferred)
{
    auto t = tensorType(operands[0].getType());
    inferred.push_back(withProperties(
        t, inverseProperties(closure(propertiesOf(t), t.getShape()))));
    return success();
}

//===----------------------------------------------------------------------===//
// ulm_lam: PropertyInferenceOpInterface (operand properties only)
//===----------------------------------------------------------------------===//

::mlir::ulm_lam::Properties
MulOp::inferResultProperties(unsigned, ArrayRef<::mlir::ulm_lam::Properties>)
{
    return mulProperties(getFactors());
}
::mlir::ulm_lam::Properties
AddOp::inferResultProperties(unsigned, ArrayRef<::mlir::ulm_lam::Properties>)
{
    return addProperties(getTerms());
}
::mlir::ulm_lam::Properties
ScaleOp::inferResultProperties(
    unsigned, ArrayRef<::mlir::ulm_lam::Properties> operandProps)
{
    return scaleProperties(getAlpha(), operandProps.back());
}
::mlir::ulm_lam::Properties
TransposeOp::inferResultProperties(unsigned,
                                   ArrayRef<::mlir::ulm_lam::Properties> ps)
{
    return transposeProperties(ps.front());
}
::mlir::ulm_lam::Properties
InverseOp::inferResultProperties(unsigned,
                                 ArrayRef<::mlir::ulm_lam::Properties> ps)
{
    return inverseProperties(ps.front());
}
::mlir::ulm_lam::Properties
IdentityOp::inferResultProperties(unsigned,
                                  ArrayRef<::mlir::ulm_lam::Properties>)
{
    return P::identity;
}
::mlir::ulm_lam::Properties
AssumeOp::inferResultProperties(unsigned, ArrayRef<::mlir::ulm_lam::Properties>)
{
    return propertiesOf(getType());
}

//===----------------------------------------------------------------------===//
// ulm_lam: verifiers
//===----------------------------------------------------------------------===//

LogicalResult
MulOp::verify()
{
    if (getFactors().size() < 2)
	return emitOpError("needs at least two factors");
    for (unsigned i = 0; i + 1 < getFactors().size(); ++i) {
	Type a = getFactors()[i].getType(), b = getFactors()[i + 1].getType();
	if (!dimsCompatible(cols(a), rows(b)))
	    return emitOpError("factor ")
	           << i << " has " << cols(a) << " columns but factor " << i + 1
	           << " has " << rows(b) << " rows";
    }
    return success();
}

LogicalResult
AddOp::verify()
{
    if (getTerms().size() < 2)
	return emitOpError("needs at least two terms");
    Type t0 = getTerms().front().getType();
    for (Value v : getTerms())
	if (!dimsCompatible(rows(v.getType()), rows(t0)) ||
	    !dimsCompatible(cols(v.getType()), cols(t0)))
	    return emitOpError("terms have different shapes");
    return success();
}

LogicalResult
InverseOp::verify()
{
    Type t = getOperand().getType();
    if (!dimsCompatible(rows(t), cols(t)))
	return emitOpError("operand is not square");
    return success();
}

LogicalResult
AssumeOp::verify()
{
    auto a = tensorType(getOperand().getType()), r = tensorType(getType());
    if (a.getShape() != r.getShape() ||
        a.getElementType() != r.getElementType())
	return emitOpError("must not change shape or element type");
    return success();
}

//===----------------------------------------------------------------------===//
// ulm_lam: canonicalisation (normal form, part 1)
//===----------------------------------------------------------------------===//

namespace {

/// Replace `op` by `v`, keeping the old result type (a stronger inferred
/// type is cast back with ulm_lam.assume, which only drops properties).
void
replaceKeepingType(PatternRewriter &rewriter, Operation *op, Value v)
{
    Type old = op->getResult(0).getType();
    if (v.getType() != old)
	v = AssumeOp::create(rewriter, op->getLoc(), old, v, /*checked=*/false);
    rewriter.replaceOp(op, v);
}

/// Flatten nested n-ary ops: mul(a, mul(b, c)) -> mul(a, b, c).
template <typename OpT> struct Flatten : OpRewritePattern<OpT>
{
    using OpRewritePattern<OpT>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
    {
	SmallVector<Value> operands;
	bool changed = false;
	for (Value v : op->getOperands()) {
	    auto inner = v.template getDefiningOp<OpT>();
	    if (inner && inner->hasOneUse()) {
		llvm::append_range(operands, inner->getOperands());
		changed = true;
	    } else {
		operands.push_back(v);
	    }
	}
	if (!changed)
	    return failure();
	auto flat = OpT::create(rewriter, op.getLoc(), operands);
	replaceKeepingType(rewriter, op, flat);
	return success();
    }
};

/// transpose(transpose(x)) -> x ; transpose(x) -> x for symmetric x.
struct FoldTranspose : OpRewritePattern<TransposeOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override
    {
	Value x = op.getOperand();
	if (auto inner = x.getDefiningOp<TransposeOp>()) {
	    replaceKeepingType(rewriter, op, inner.getOperand());
	    return success();
	}
	auto t = tensorType(x.getType());
	if (has(closure(propertiesOf(t), t.getShape()), P::symmetric)) {
	    replaceKeepingType(rewriter, op, x);
	    return success();
	}
	return failure();
    }
};

/// inverse(inverse(x)) -> x.
struct FoldInverse : OpRewritePattern<InverseOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(InverseOp op, PatternRewriter &rewriter) const override
    {
	auto inner = op.getOperand().getDefiningOp<InverseOp>();
	if (!inner)
	    return failure();
	replaceKeepingType(rewriter, op, inner.getOperand());
	return success();
    }
};

/// scale(1, x) -> x.
struct FoldScaleOne : OpRewritePattern<ScaleOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override
    {
	FloatAttr c;
	if (!matchPattern(op.getAlpha(), m_Constant(&c)) ||
	    c.getValueAsDouble() != 1.0)
	    return failure();
	replaceKeepingType(rewriter, op, op.getOperand());
	return success();
    }
};

/// a (b X) -> (a b) X for constants a and b (e.g. -(3 X), from A - 3 X).
struct FoldScaleScale : OpRewritePattern<ScaleOp>
{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override
    {
	auto inner = op.getOperand().getDefiningOp<ScaleOp>();
	FloatAttr a, b;
	if (!inner || !matchPattern(op.getAlpha(), m_Constant(&a)) ||
	    !matchPattern(inner.getAlpha(), m_Constant(&b)))
	    return failure();
	Value ab = arith::ConstantOp::create(
	    rewriter, op.getLoc(), a.getType(),
	    rewriter.getFloatAttr(a.getType(),
	                          a.getValueAsDouble() * b.getValueAsDouble()));
	rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getType(), ab,
	                                     inner.getOperand());
	return success();
    }
};

} // namespace

void
MulOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *ctx)
{
    results.add<Flatten<MulOp>>(ctx);
}
void
AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *ctx)
{
    results.add<Flatten<AddOp>>(ctx);
}
void
TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *ctx)
{
    results.add<FoldTranspose>(ctx);
}
void
InverseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *ctx)
{
    results.add<FoldInverse>(ctx);
}
void
ScaleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                     MLIRContext *ctx)
{
    results.add<FoldScaleOne, FoldScaleScale>(ctx);
}

//===----------------------------------------------------------------------===//
// ulm_lamk: verifiers, effects, properties, costs
//===----------------------------------------------------------------------===//

namespace {

using EffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

/// Memref operands are read; DPS inits are also written.  Tensor
/// operands have value semantics (no effects).
void
dpsEffects(Operation *op, EffectList &effects)
{
    auto dps = cast<DestinationStyleOpInterface>(op);
    for (OpOperand &o : op->getOpOperands()) {
	if (!isa<MemRefType>(o.get().getType()))
	    continue;
	effects.emplace_back(MemoryEffects::Read::get(), &o,
	                     SideEffects::DefaultResource::get());
	if (dps.isDpsInit(&o))
	    effects.emplace_back(MemoryEffects::Write::get(), &o,
	                         SideEffects::DefaultResource::get());
    }
}

/// Tensor inits <-> results, one to one with equal types.
LogicalResult
verifyDps(Operation *op)
{
    auto dps = cast<DestinationStyleOpInterface>(op);
    unsigned tensorInits = 0;
    for (OpOperand &o : dps.getDpsInitsMutable())
	if (isa<RankedTensorType>(o.get().getType())) {
	    if (tensorInits >= op->getNumResults() ||
	        op->getResult(tensorInits).getType() != o.get().getType())
		return op->emitOpError("result ")
		       << tensorInits << " must have the type of its init";
	    ++tensorInits;
	}
    // further results are scalars (the info of a factorisation)
    unsigned tensorResults = llvm::count_if(
        op->getResultTypes(), [](Type t) { return isa<TensorType>(t); });
    if (tensorInits != tensorResults)
	return op->emitOpError("expects one result per tensor init");
    return success();
}

std::array<int64_t, 2>
shapeOf(Value v)
{
    auto t = cast<ShapedType>(v.getType());
    return {t.getDimSize(0), t.getDimSize(1)};
}

bool
anyDynamic(std::initializer_list<int64_t> ds)
{
    return llvm::any_of(ds, [](int64_t d) { return ShapedType::isDynamic(d); });
}

double
bytesOf(Operation *op)
{
    double b = 0;
    for (Value v : op->getOperands())
	if (auto t = dyn_cast<ShapedType>(v.getType()); t && t.hasStaticShape())
	    b += t.getNumElements() * (t.getElementTypeBitWidth() / 8);
    return b;
}

} // namespace

#define ULM_LAMK_COMMON(OP)                                                    \
    void mlir::ulm_lamk::OP::getEffects(EffectList &effects)                   \
    {                                                                          \
	dpsEffects(getOperation(), effects);                                   \
    }                                                                          \
    FailureOr<double> mlir::ulm_lamk::OP::estimateBytes(                       \
        const SizeAssignment &)                                                \
    {                                                                          \
	return bytesOf(getOperation());                                        \
    }
ULM_LAMK_COMMON(GemmOp)
ULM_LAMK_COMMON(SymmOp)
ULM_LAMK_COMMON(TrmmOp)
ULM_LAMK_COMMON(TrsmOp)
ULM_LAMK_COMMON(SyrkOp)
ULM_LAMK_COMMON(GemvOp)
ULM_LAMK_COMMON(TrsvOp)
ULM_LAMK_COMMON(PotrfOp)
ULM_LAMK_COMMON(GetrfOp)
ULM_LAMK_COMMON(GetrsOp)
ULM_LAMK_COMMON(GeaxpyOp)
ULM_LAMK_COMMON(GescalOp)
ULM_LAMK_COMMON(SymvOp)
ULM_LAMK_COMMON(SycompleteOp)
#undef ULM_LAMK_COMMON

#define ULM_LAMK_GENERAL(OP)                                                   \
    ::mlir::ulm_lam::Properties mlir::ulm_lamk::OP::inferResultProperties(     \
        unsigned, ArrayRef<::mlir::ulm_lam::Properties>)                       \
    {                                                                          \
	return P::general;                                                     \
    }
ULM_LAMK_GENERAL(GemmOp)
ULM_LAMK_GENERAL(SymmOp)
ULM_LAMK_GENERAL(TrmmOp)
ULM_LAMK_GENERAL(GemvOp)
ULM_LAMK_GENERAL(TrsvOp)
ULM_LAMK_GENERAL(PotrfOp) // only one triangle is meaningful
ULM_LAMK_GENERAL(GetrsOp)
ULM_LAMK_GENERAL(GeaxpyOp)
ULM_LAMK_GENERAL(GescalOp)
ULM_LAMK_GENERAL(SymvOp)
#undef ULM_LAMK_GENERAL

::mlir::ulm_lam::Properties
mlir::ulm_lamk::TrsmOp::inferResultProperties(
    unsigned, ArrayRef<::mlir::ulm_lam::Properties> ps)
{
    // L^-1 B keeps the rank of B (operands: alpha, A, B)
    return has(ps.back(), P::full_rank) ? P::full_rank : P::general;
}
::mlir::ulm_lam::Properties
mlir::ulm_lamk::SyrkOp::inferResultProperties(
    unsigned, ArrayRef<::mlir::ulm_lam::Properties>)
{
    return P::spsd;
}

#define ULM_LAMK_VERIFY_DPS(OP)                                                \
    LogicalResult mlir::ulm_lamk::OP::verify()                                 \
    {                                                                          \
	return verifyDps(getOperation());                                      \
    }
ULM_LAMK_VERIFY_DPS(SymmOp)
ULM_LAMK_VERIFY_DPS(TrmmOp)
ULM_LAMK_VERIFY_DPS(TrsmOp)
ULM_LAMK_VERIFY_DPS(SyrkOp)
ULM_LAMK_VERIFY_DPS(GemvOp)
ULM_LAMK_VERIFY_DPS(TrsvOp)
ULM_LAMK_VERIFY_DPS(PotrfOp)
ULM_LAMK_VERIFY_DPS(GetrsOp)
ULM_LAMK_VERIFY_DPS(GeaxpyOp)
ULM_LAMK_VERIFY_DPS(GescalOp)
ULM_LAMK_VERIFY_DPS(SymvOp)
ULM_LAMK_VERIFY_DPS(SycompleteOp)
#undef ULM_LAMK_VERIFY_DPS

LogicalResult
mlir::ulm_lamk::GemmOp::verify()
{
    if (failed(verifyDps(getOperation())))
	return failure();
    auto [ar, ac] = shapeOf(getA());
    auto [br, bc] = shapeOf(getB());
    auto [cr, cc] = shapeOf(getC());
    int64_t m = getTransA() == Trans::trans ? ac : ar;
    int64_t k = getTransA() == Trans::trans ? ar : ac;
    int64_t k2 = getTransB() == Trans::trans ? bc : br;
    int64_t n = getTransB() == Trans::trans ? br : bc;
    if (!dimsCompatible(k, k2) || !dimsCompatible(m, cr) ||
        !dimsCompatible(n, cc))
	return emitOpError("shapes of op(A), op(B) and C do not match");
    return success();
}

FailureOr<double>
mlir::ulm_lamk::GemmOp::estimateFlops(const SizeAssignment &)
{
    auto [cr, cc] = shapeOf(getC());
    auto [ar, ac] = shapeOf(getA());
    int64_t k = getTransA() == Trans::trans ? ar : ac;
    if (anyDynamic({cr, cc, k}))
	return failure();
    return 2.0 * cr * cc * k;
}
FailureOr<double>
mlir::ulm_lamk::SymmOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getC());
    if (anyDynamic({m, n}))
	return failure();
    int64_t s = getSide() == Side::left ? m : n;
    return 2.0 * s * m * n;
}
FailureOr<double>
mlir::ulm_lamk::TrmmOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getB());
    if (anyDynamic({m, n}))
	return failure();
    int64_t s = getSide() == Side::left ? m : n;
    return 1.0 * s * m * n;
}
FailureOr<double>
mlir::ulm_lamk::TrsmOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getB());
    if (anyDynamic({m, n}))
	return failure();
    int64_t s = getSide() == Side::left ? m : n;
    return 1.0 * s * m * n;
}
FailureOr<double>
mlir::ulm_lamk::SyrkOp::estimateFlops(const SizeAssignment &)
{
    auto [n, n2] = shapeOf(getC());
    auto [ar, ac] = shapeOf(getA());
    int64_t k = getTrans() == Trans::trans ? ar : ac;
    if (anyDynamic({n, k}))
	return failure();
    return 1.0 * n * n * k;
}
FailureOr<double>
mlir::ulm_lamk::GemvOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getA());
    if (anyDynamic({m, n}))
	return failure();
    return 2.0 * m * n;
}
FailureOr<double>
mlir::ulm_lamk::TrsvOp::estimateFlops(const SizeAssignment &)
{
    auto [n, n2] = shapeOf(getA());
    if (anyDynamic({n}))
	return failure();
    return 1.0 * n * n;
}
FailureOr<double>
mlir::ulm_lamk::PotrfOp::estimateFlops(const SizeAssignment &)
{
    auto [n, n2] = shapeOf(getA());
    if (anyDynamic({n}))
	return failure();
    return 1.0 * n * n * n / 3.0;
}
FailureOr<double>
mlir::ulm_lamk::GetrfOp::estimateFlops(const SizeAssignment &)
{
    auto [n, n2] = shapeOf(getA());
    if (anyDynamic({n}))
	return failure();
    return 2.0 * n * n * n / 3.0;
}
FailureOr<double>
mlir::ulm_lamk::GetrsOp::estimateFlops(const SizeAssignment &)
{
    auto [n, n2] = shapeOf(getLU());
    auto [br, bc] = shapeOf(getB());
    if (anyDynamic({n, bc}))
	return failure();
    return 2.0 * n * n * bc;
}
FailureOr<double>
mlir::ulm_lamk::GeaxpyOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getY());
    if (anyDynamic({m, n}))
	return failure();
    return 2.0 * m * n;
}
FailureOr<double>
mlir::ulm_lamk::GescalOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getX());
    if (anyDynamic({m, n}))
	return failure();
    return 1.0 * m * n;
}
FailureOr<double>
mlir::ulm_lamk::SymvOp::estimateFlops(const SizeAssignment &)
{
    auto [m, n] = shapeOf(getA());
    if (anyDynamic({m, n}))
	return failure();
    return 2.0 * m * n;
}
FailureOr<double>
mlir::ulm_lamk::SycompleteOp::estimateFlops(const SizeAssignment &)
{
    return 0.0; // memory traffic only (estimateBytes)
}
::mlir::ulm_lam::Properties
mlir::ulm_lamk::SycompleteOp::inferResultProperties(
    unsigned, ArrayRef<::mlir::ulm_lam::Properties> ps)
{
    // the same matrix, now stored completely
    return (ps.empty() ? P::symmetric : ps.back() | P::symmetric) &
           ~(P::lower_stored | P::upper_stored);
}

/// The callee is unknown: besides its operands it may read and write any
/// memory (and do I/O), so the call is never hoisted, duplicated or removed.
void
mlir::ulm_lamk::CallOp::getEffects(EffectList &effects)
{
    dpsEffects(getOperation(), effects);
    effects.emplace_back(MemoryEffects::Read::get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ulm_lamk: bufferization
//===----------------------------------------------------------------------===//

namespace {

/// Bufferize a ulm_lamk op in destination-passing style: the same op on
/// buffers, no results; tensor results are replaced by their init buffers.
/// Analysis (in-place decisions, copies) comes from the DPS default model.
template <typename OpT>
struct UlmLamkBufferizableModel
    : bufferization::DstBufferizableOpInterfaceExternalModel<
          UlmLamkBufferizableModel<OpT>, OpT>
{
    LogicalResult
    bufferize(Operation *op, RewriterBase &rewriter,
              const bufferization::BufferizationOptions &options,
              bufferization::BufferizationState &state) const
    {
	auto dps = cast<DestinationStyleOpInterface>(op);
	if (dps.hasPureBufferSemantics())
	    return success();
	SmallVector<Value> operands, initBuffers;
	for (OpOperand &o : op->getOpOperands()) {
	    Value v = o.get();
	    if (isa<TensorType>(v.getType())) {
		FailureOr<Value> buf =
		    bufferization::getBuffer(rewriter, v, options, state);
		if (failed(buf))
		    return failure();
		v = *buf;
		if (dps.isDpsInit(&o))
		    initBuffers.push_back(v);
	    }
	    operands.push_back(v);
	}
	rewriter.setInsertionPoint(op);
	OperationState s(op->getLoc(), op->getName());
	s.addOperands(operands);
	s.addAttributes(llvm::to_vector(op->getDiscardableAttrs()));
	s.propertiesAttr = op->getPropertiesAsAttribute();
	// results that are not tensors (the info of a factorisation) stay
	for (Value r : op->getResults())
	    if (!isa<TensorType>(r.getType()))
		s.addTypes(r.getType());
	Operation *b = rewriter.create(s);
	SmallVector<Value> values(initBuffers);
	llvm::append_range(values, b->getResults());
	bufferization::replaceOpWithBufferizedValues(rewriter, op, values);
	return success();
    }
};

template <typename... Ops>
void
attachAll(MLIRContext *ctx)
{
    (Ops::template attachInterface<UlmLamkBufferizableModel<Ops>>(*ctx), ...);
}

} // namespace

void
mlir::ulm_lam::registerBufferizationModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, ulm_lamk::UlmLamkDialect *) {
	using namespace mlir::ulm_lamk;
	attachAll<GemmOp, SymmOp, TrmmOp, TrsmOp, SyrkOp, GemvOp, SymvOp,
	          SycompleteOp, TrsvOp, PotrfOp, GetrfOp, GetrsOp, GeaxpyOp,
	          GescalOp, CallOp>(ctx);
    });
}

//===----------------------------------------------------------------------===//
// Dynamic dimensions
//===----------------------------------------------------------------------===//

namespace {

/// Position of dimension `i` among the dynamic dimensions of `t`.
unsigned
dynamicIndex(ShapedType t, unsigned i)
{
    unsigned k = 0;
    for (unsigned d = 0; d < i; ++d)
	k += ShapedType::isDynamic(t.getDimSize(d));
    return k;
}

/// Dimension `i` of a value declared with `ulm_lam.dims = [...]`: the
/// corresponding operand of `args`.
Value
fromDimsAttr(Attribute attr, ShapedType t, unsigned i, ValueRange args)
{
    auto dims = dyn_cast_if_present<ArrayAttr>(attr);
    unsigned k = dynamicIndex(t, i);
    if (!dims || k >= dims.size())
	return {};
    auto pos = dyn_cast<IntegerAttr>(dims[k]);
    if (!pos || pos.getInt() < 0 ||
        static_cast<size_t>(pos.getInt()) >= args.size())
	return {};
    return args[pos.getInt()];
}

} // namespace

OpFoldResult
mlir::ulm_lam::dimOf(Value v, unsigned i)
{
    auto t = dyn_cast<ShapedType>(v.getType());
    if (!t || i >= static_cast<unsigned>(t.getRank()))
	return {};
    if (!t.isDynamicDim(i))
	return IntegerAttr::get(IndexType::get(v.getContext()),
	                        t.getDimSize(i));
    if (auto arg = dyn_cast<BlockArgument>(v)) {
	Operation *parent = arg.getOwner()->getParentOp();
	if (!parent)
	    return {};
	if (auto f = dyn_cast<FunctionOpInterface>(parent))
	    return fromDimsAttr(f.getArgAttr(arg.getArgNumber(), kDimsAttr), t,
	                        i, arg.getOwner()->getArguments());
	if (auto loop = dyn_cast<scf::ForOp>(parent))
	    if (OpOperand *init = loop.getTiedLoopInit(arg))
		return dimOf(init->get(), i);
	return {};
    }
    Operation *def = v.getDefiningOp();
    unsigned r = cast<OpResult>(v).getResultNumber();
    // ulm_lam ops: from their operands
    if (auto m = dyn_cast<MulOp>(def)) {
	Value f = i == 0 ? m.getFactors().front() : m.getFactors().back();
	return dimOf(f, i);
    }
    if (auto a = dyn_cast<AddOp>(def)) {
	for (Value term : a.getTerms())
	    if (OpFoldResult d = dimOf(term, i))
		return d;
	return {};
    }
    if (auto tr = dyn_cast<TransposeOp>(def))
	return dimOf(tr.getOperand(), 1 - i);
    if (auto inv = dyn_cast<InverseOp>(def))
	return dimOf(inv.getOperand(), i);
    if (auto sc = dyn_cast<ScaleOp>(def))
	return dimOf(sc.getOperand(), i);
    if (auto as = dyn_cast<AssumeOp>(def))
	return dimOf(as.getOperand(), i);
    if (auto id = dyn_cast<IdentityOp>(def))
	return id.getDynamicSizes()[dynamicIndex(t, i)];
    // allocations
    if (auto e = dyn_cast<tensor::EmptyOp>(def))
	return e.getDynamicSize(i);
    if (auto a = dyn_cast<memref::AllocOp>(def))
	return a.getDynamicSizes()[dynamicIndex(t, i)];
    if (auto a = dyn_cast<memref::AllocaOp>(def))
	return a.getDynamicSizes()[dynamicIndex(t, i)];
    if (auto vw = dyn_cast<memref::ViewOp>(def))
	return vw.getSizes()[dynamicIndex(t, i)];
    // the same matrix under another type
    if (auto c = dyn_cast<memref::CastOp>(def))
	return dimOf(c.getSource(), i);
    if (auto c = dyn_cast<tensor::CastOp>(def))
	return dimOf(c.getSource(), i);
    if (auto c = dyn_cast<bufferization::ToTensorOp>(def))
	return dimOf(c.getBuffer(), i);
    if (auto c = dyn_cast<bufferization::ToBufferOp>(def))
	return dimOf(c.getTensor(), i);
    if (auto m = dyn_cast<bufferization::MaterializeInDestinationOp>(def))
	return dimOf(m.getDest(), i);
    // destination-passing style: the destination
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(def))
	if (r < dps.getNumDpsInits())
	    return dimOf(dps.getDpsInits()[r], i);
    // loops and branches: the value that enters / is yielded
    if (auto loop = dyn_cast<scf::ForOp>(def))
	return dimOf(loop.getInits()[r], i);
    if (auto br = dyn_cast<scf::IfOp>(def)) {
	if (OpFoldResult d = dimOf(br.thenYield().getOperand(r), i))
	    return d;
	if (!br.getElseRegion().empty())
	    return dimOf(br.elseYield().getOperand(r), i);
	return {};
    }
    // a call: the callee's result attribute names the call operand
    if (auto call = dyn_cast<func::CallOp>(def)) {
	auto callee = dyn_cast_if_present<func::FuncOp>(
	    SymbolTable::lookupNearestSymbolFrom(def, call.getCalleeAttr()));
	if (!callee)
	    return {};
	return fromDimsAttr(callee.getResultAttr(r, kDimsAttr), t, i,
	                    call.getOperands());
    }
    return {};
}

Value
mlir::ulm_lam::dimValue(OpBuilder &b, Location loc, Value v, unsigned i)
{
    OpFoldResult d = dimOf(v, i);
    if (!d)
	return {};
    if (auto a = dyn_cast<Attribute>(d))
	return arith::ConstantIndexOp::create(b, loc,
	                                      cast<IntegerAttr>(a).getInt());
    return cast<Value>(d);
}

FailureOr<SmallVector<Value>>
mlir::ulm_lam::dynamicSizes(OpBuilder &b, Location loc, ShapedType type,
                            Value rowsOf, unsigned rowsDim, Value colsOf,
                            unsigned colsDim)
{
    SmallVector<Value> sizes;
    for (unsigned d = 0; d < 2; ++d) {
	if (!type.isDynamicDim(d))
	    continue;
	Value s = d == 0 ? dimValue(b, loc, rowsOf, rowsDim)
	                 : dimValue(b, loc, colsOf, colsDim);
	if (!s)
	    return failure();
	sizes.push_back(s);
    }
    return sizes;
}

void
mlir::ulm_lam::resolveDimOps(Operation *root)
{
    SmallVector<std::pair<Operation *, std::pair<Value, int64_t>>> dims;
    root->walk([&](Operation *op) {
	std::optional<int64_t> i;
	Value src;
	if (auto d = dyn_cast<memref::DimOp>(op)) {
	    i = d.getConstantIndex();
	    src = d.getSource();
	} else if (auto d = dyn_cast<tensor::DimOp>(op)) {
	    i = d.getConstantIndex();
	    src = d.getSource();
	}
	if (i && *i >= 0)
	    dims.push_back({op, {src, *i}});
    });
    for (auto &[op, d] : dims) {
	OpFoldResult r = dimOf(d.first, d.second);
	if (!r)
	    continue;
	Value v;
	if (auto a = dyn_cast<Attribute>(r)) {
	    OpBuilder b(op);
	    v = arith::ConstantIndexOp::create(b, op->getLoc(),
	                                       cast<IntegerAttr>(a).getInt());
	} else {
	    v = cast<Value>(r);
	}
	op->getResult(0).replaceAllUsesWith(v);
	op->erase();
    }
}
