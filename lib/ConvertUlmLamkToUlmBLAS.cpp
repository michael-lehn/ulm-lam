//===- ConvertUlmLamkToUlmBLAS.cpp - lower bufferized kernels to ulmBLAS --===//
//
// convert-ulm-lamk-to-ulmblas: lower the bufferized program to EmitC so that
// `mlir-translate --mlir-to-cpp` prints C++17 calling the header-only
// ulmBLAS templates (ulmBLAS::gemm, gemv, trlsm, trusm, trlsv, trusv,
// geaxpy, gescal, potrf, getrf, getrs).
//
// The ulm_lam ABI:
//   * every statically shaped, identity-layout memref is a row-major
//     pointer (`double *`, `float *`, pivots `int32_t *`); in ulmBLAS terms
//     the strides are (incRow, incCol) = (columns, 1).  So
//     `foo(index, memref<500x500xf64>)` is `void foo(size_t, double *)`;
//   * index is size_t;
//   * functions get `extern "C"`, so C code can call the generated C++ and
//     the generated C++ can call C functions;
//   * temporaries come from a caller-provided workspace (ulm-lam-workspace):
//     `int8_t *work, int64_t *workspace_info` as the last two parameters,
//     LAPACK-style (-1: query, 0: allocate internally, > 0: size given);
//   * matrices are passed by reference: a function with a matrix output
//     receives a pointer to it, and the caller allocates and frees the
//     matrix (the caller owns it).  No function returns a matrix, so no
//     ownership ever moves from callee to caller.  The pass rejects a
//     function that still returns one (ulm-lam-results-to-out-params and
//     buffer-results-to-out-params establish the convention).
//
// ulmBLAS takes strides instead of flags: a transposed operand is the same
// pointer with swapped strides, the transpose of a lower triangular matrix
// is an upper triangular one, and a triangular solve from the right is a
// solve from the left on transposed views.  So ulm_lamk's trans and side
// attributes disappear in the lowering, and so does row-major versus
// column-major.
//
// Memory: memref.alloc -> aligned_alloc (size rounded up to the alignment,
// as C11/C++17 require), memref.dealloc -> free, memref.copy -> memcpy.
// Control flow, arithmetic and functions use the upstream
// {scf,arith,func}-to-emitc patterns with this type converter.
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ScopeExit.h"

#include <array>
#include <string>
#include <vector>

using namespace mlir;
namespace lk = mlir::ulm_lamk;

namespace {

constexpr int64_t kAlignment = 64;

/// Identity layout, f32/f64 (rank 1/2) or i32 pivots (rank 1); sizes may
/// be dynamic (they are then function parameters, see ulm_lam::dimOf).
bool
isCArray(MemRefType t)
{
    if (!t || !t.getLayout().isIdentity())
	return false;
    Type e = t.getElementType();
    // the workspace: a byte buffer of any size
    if (e.isInteger(8))
	return t.getRank() == 1;
    if (e.isF64() || e.isF32())
	return t.getRank() == 1 || t.getRank() == 2;
    // pivots, and *workspace_info
    return (e.isInteger(32) || e.isInteger(64)) && t.getRank() == 1;
}

int64_t
byteSize(MemRefType t)
{
    return t.getNumElements() * (t.getElementTypeBitWidth() / 8);
}

emitc::PointerType
voidPtr(MLIRContext *ctx)
{
    return emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
}

/// The dynamic dimensions of the memref operands of kernels and copies,
/// found before the conversion (which moves blocks, so that dimOf cannot
/// trace values any more): (memref, dim) -> index value or constant.
using DimTable = llvm::DenseMap<std::pair<Value, unsigned>, OpFoldResult>;
DimTable *dimTable = nullptr; // set while the pass runs

void
collectDims(ModuleOp module, DimTable &table)
{
    module.walk([&](Operation *op) {
	if (!isa<lk::UlmLamkDialect>(op->getDialect()) &&
	    !isa<memref::CopyOp>(op))
	    return;
	for (Value v : op->getOperands())
	    if (auto t = dyn_cast<MemRefType>(v.getType()))
		for (unsigned d = 0, r = t.getRank(); d < r; ++d)
		    if (t.isDynamicDim(d))
			table[{v, d}] = ulm_lam::dimOf(v, d);
    });
}

OpFoldResult
lookupDim(Value v, unsigned d)
{
    auto t = cast<MemRefType>(v.getType());
    if (!t.isDynamicDim(d))
	return IntegerAttr::get(IndexType::get(v.getContext()),
	                        t.getDimSize(d));
    if (!dimTable)
	return {};
    return dimTable->lookup({v, d});
}

// ---- memory -------------------------------------------------------------

struct AllocLowering : OpConversionPattern<memref::AllocOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
	MemRefType t = op.getType();
	if (!isCArray(t) ||
	    (!t.hasStaticShape() && !t.getElementType().isInteger(8)))
	    return rewriter.notifyMatchFailure(
	        op, "not a static C array or a byte buffer");
	MLIRContext *ctx = op.getContext();
	int64_t align = std::max<int64_t>(
	    op.getAlignment().value_or(kAlignment), kAlignment);
	emitc::CallOpaqueOp call;
	if (t.hasStaticShape()) {
	    int64_t bytes = (byteSize(t) + align - 1) / align * align;
	    call = emitc::CallOpaqueOp::create(
	        rewriter, op.getLoc(), TypeRange{voidPtr(ctx)}, "aligned_alloc",
	        ValueRange{},
	        rewriter.getArrayAttr({rewriter.getI64IntegerAttr(align),
		                       rewriter.getI64IntegerAttr(bytes)}));
	} else {
	    // a byte buffer of run-time size (the workspace; ulm-lam-workspace
	    // makes it a multiple of the alignment)
	    call = emitc::CallOpaqueOp::create(
	        rewriter, op.getLoc(), TypeRange{voidPtr(ctx)}, "aligned_alloc",
	        adaptor.getDynamicSizes(),
	        rewriter.getArrayAttr({rewriter.getI64IntegerAttr(align),
		                       rewriter.getIndexAttr(0)}));
	}
	Type ptr = getTypeConverter()->convertType(t);
	rewriter.replaceOpWithNewOp<emitc::CastOp>(op, ptr, call.getResult(0));
	return success();
    }
};

struct DeallocLowering : OpConversionPattern<memref::DeallocOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
	if (!isCArray(dyn_cast<MemRefType>(op.getMemref().getType())))
	    return rewriter.notifyMatchFailure(op, "not a static C array");
	rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
	    op, TypeRange{}, "free", ValueRange{adaptor.getMemref()});
	return success();
    }
};

/// x & mask on index values (size_t): the alignment of workspace parts.
/// (arith-to-emitc converts bitwise ops only on integers.)
struct IndexAndLowering : OpConversionPattern<arith::AndIOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(arith::AndIOp op, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	if (!op.getType().isIndex())
	    return rw.notifyMatchFailure(op, "not on index");
	rw.replaceOpWithNewOp<emitc::BitwiseAndOp>(
	    op, getTypeConverter()->convertType(op.getType()), a.getLhs(),
	    a.getRhs());
	return success();
    }
};

// ---- the workspace ------------------------------------------------------

/// A temporary in the workspace: (T *)(work + offset).
struct ViewLowering : OpConversionPattern<memref::ViewOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::ViewOp op, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	// (the sizes are not needed: the view is a pointer)
	if (!isCArray(op.getType()))
	    return rw.notifyMatchFailure(op, "not a C array view");
	// pointer arithmetic needs an integer offset (not size_t)
	Value off = emitc::CastOp::create(rw, op.getLoc(), rw.getI64Type(),
	                                  a.getByteShift());
	Value p = emitc::AddOp::create(rw, op.getLoc(), a.getSource().getType(),
	                               a.getSource(), off);
	rw.replaceOpWithNewOp<emitc::CastOp>(
	    op, getTypeConverter()->convertType(op.getType()), p);
	return success();
    }
};

/// A cast between C arrays is the same pointer.
struct CastLowering : OpConversionPattern<memref::CastOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::CastOp op, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	if (!isCArray(dyn_cast<MemRefType>(op.getType())))
	    return rw.notifyMatchFailure(op, "not a C array");
	rw.replaceOp(op, a.getSource());
	return success();
    }
};

/// x = p[i] and p[i] = x on rank-1 C arrays (*workspace_info).
struct LoadLowering : OpConversionPattern<memref::LoadOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::LoadOp op, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	if (!isCArray(op.getMemRefType()) || op.getMemRefType().getRank() != 1)
	    return rw.notifyMatchFailure(op, "not a rank-1 C array");
	Type t = op.getType();
	auto ref = emitc::SubscriptOp::create(rw, op.getLoc(),
	                                      emitc::LValueType::get(t),
	                                      a.getMemref(), a.getIndices());
	rw.replaceOpWithNewOp<emitc::LoadOp>(op, t, ref);
	return success();
    }
};

struct StoreLowering : OpConversionPattern<memref::StoreOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::StoreOp op, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	if (!isCArray(op.getMemRefType()) || op.getMemRefType().getRank() != 1)
	    return rw.notifyMatchFailure(op, "not a rank-1 C array");
	auto ref = emitc::SubscriptOp::create(
	    rw, op.getLoc(),
	    emitc::LValueType::get(op.getValueToStore().getType()),
	    a.getMemref(), a.getIndices());
	emitc::AssignOp::create(rw, op.getLoc(), ref, a.getValue());
	rw.eraseOp(op);
	return success();
    }
};

struct CopyLowering : OpConversionPattern<memref::CopyOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
	auto t = dyn_cast<MemRefType>(op.getSource().getType());
	if (!isCArray(t) ||
	    !isCArray(dyn_cast<MemRefType>(op.getTarget().getType())))
	    return rewriter.notifyMatchFailure(op, "not a C array copy");
	if (t.hasStaticShape()) {
	    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
	        op, TypeRange{}, "memcpy",
	        ValueRange{adaptor.getTarget(), adaptor.getSource()},
	        rewriter.getArrayAttr(
	            {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
		     rewriter.getI64IntegerAttr(byteSize(t))}));
	    return success();
	}
	// bytes = element size * rows * cols, with run-time dimensions
	Location loc = op.getLoc();
	Type sizeT = emitc::SizeTType::get(op.getContext());
	Value bytes = emitc::ConstantOp::create(
	    rewriter, loc, sizeT,
	    emitc::OpaqueAttr::get(
	        op.getContext(),
	        std::to_string(t.getElementTypeBitWidth() / 8)));
	for (unsigned d = 0, r = t.getRank(); d < r; ++d) {
	    Value s;
	    if (!t.isDynamicDim(d)) {
		if (t.getDimSize(d) == 1)
		    continue;
		s = emitc::ConstantOp::create(
		    rewriter, loc, sizeT,
		    emitc::OpaqueAttr::get(op.getContext(),
		                           std::to_string(t.getDimSize(d))));
	    } else {
		OpFoldResult r = lookupDim(op.getSource(), d);
		if (!r)
		    r = lookupDim(op.getTarget(), d);
		if (!r || !isa<Value>(r))
		    return rewriter.notifyMatchFailure(op, "unknown size");
		s = rewriter.getRemappedValue(cast<Value>(r));
	    }
	    bytes = emitc::MulOp::create(rewriter, loc, sizeT, bytes, s);
	}
	rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
	    op, TypeRange{}, "memcpy",
	    ValueRange{adaptor.getTarget(), adaptor.getSource(), bytes});
	return success();
    }
};

// ---- kernels ------------------------------------------------------------

/// The index type of all ulmBLAS calls (sizes, strides): ulmBLAS deduces
/// it from the arguments, so it is named explicitly, e.g.
/// ulmBLAS::gemm<std::ptrdiff_t>(...).
#define kIndexType "<std::ptrdiff_t>"

/// A size or stride argument: a literal, or a value (a dynamic dimension).
struct Dim
{
    int64_t c = 0;
    Value v; // converted (size_t) value if dynamic
};

/// op(M) of a row-major matrix (or n x 1 vector) operand in ulmBLAS terms.
struct View
{
    Dim rows, cols;     // of op(M)
    Dim incRow, incCol; // strides of op(M)
};

/// Dimension `d` of the (original, not yet converted) memref `v`.
Dim
dimArg(ConversionPatternRewriter &rw, Value v, unsigned d)
{
    auto t = cast<MemRefType>(v.getType());
    if (d >= static_cast<unsigned>(t.getRank()))
	return {1, {}};
    OpFoldResult r = lookupDim(v, d);
    if (auto a = dyn_cast_if_present<Attribute>(r))
	return {cast<IntegerAttr>(a).getInt(), {}};
    if (!r)
	return {-1, {}};
    return {0, rw.getRemappedValue(cast<Value>(r))};
}

View
view(ConversionPatternRewriter &rw, Value v, bool trans = false)
{
    Dim r = dimArg(rw, v, 0), c = dimArg(rw, v, 1);
    if (trans)
	return {c, r, {1, {}}, c};
    return {r, c, c, {1, {}}};
}

/// The arguments of a call: operands are referenced by position, the
/// others are literals.
struct Args
{
    ConversionPatternRewriter &rw;
    SmallVector<Value> ops;
    SmallVector<Attribute> attrs;

    Args &
    op(Value v)
    {
	attrs.push_back(rw.getIndexAttr(ops.size()));
	ops.push_back(v);
	return *this;
    }
    /// Sizes and strides.  The calls name IndexType = std::ptrdiff_t
    /// explicitly (see kIndexType), so literals and size_t parameters
    /// convert to it.
    Args &
    dim(const Dim &d)
    {
	if (d.v)
	    return op(d.v);
	attrs.push_back(rw.getI32IntegerAttr(d.c));
	return *this;
    }
    Args &
    lit(int64_t v)
    {
	attrs.push_back(rw.getI32IntegerAttr(v));
	return *this;
    }
    Args &
    flag(bool b)
    {
	attrs.push_back(
	    emitc::OpaqueAttr::get(rw.getContext(), b ? "true" : "false"));
	return *this;
    }
    Args &
    strides(const View &m)
    {
	return dim(m.incRow).dim(m.incCol);
    }
    LogicalResult
    call(Operation *o, StringRef callee)
    {
	rw.replaceOpWithNewOp<emitc::CallOpaqueOp>(
	    o, TypeRange{}, callee, ValueRange(ops), rw.getArrayAttr(attrs));
	return success();
    }
};

/// All shaped operands must be row-major C arrays with known sizes.
bool
allCArrays(Operation *o)
{
    return llvm::all_of(o->getOperands(), [](Value v) {
	auto t = dyn_cast<MemRefType>(v.getType());
	if (!isa<ShapedType>(v.getType()))
	    return true;
	if (!isCArray(t))
	    return false;
	for (unsigned d = 0, r = t.getRank(); d < r; ++d)
	    if (!lookupDim(v, d))
		return false;
	return true;
    });
}

#define ULM_LAMK_LOWERING(NAME, OP)                                            \
    struct NAME : OpConversionPattern<OP>                                      \
    {                                                                          \
	using OpConversionPattern::OpConversionPattern;                        \
	using Rewriter = ConversionPatternRewriter;                            \
	LogicalResult                                                          \
	matchAndRewrite(OP o, OpAdaptor a, Rewriter &rw) const override        \
	{                                                                      \
	    if (!allCArrays(o))                                                \
		return rw.notifyMatchFailure(                                  \
		    o, "not on row-major arrays of known size");               \
	    return lower(o, a, rw);                                            \
	}                                                                      \
	static LogicalResult                                                   \
	lower(OP o, OpAdaptor a, Rewriter &rw);                                \
    };                                                                         \
    LogicalResult NAME::lower(OP o, OpAdaptor a, Rewriter &rw)

// C := alpha op(A) op(B) + beta C
ULM_LAMK_LOWERING(GemmLowering, lk::GemmOp)
{
    View A = view(rw, o.getA(), o.getTransA() == lk::Trans::trans);
    View B = view(rw, o.getB(), o.getTransB() == lk::Trans::trans);
    View C = view(rw, o.getC());
    return Args{rw}
        .dim(C.rows)
        .dim(C.cols)
        .dim(A.cols)
        .op(a.getAlpha())
        .flag(false)
        .op(a.getA())
        .strides(A)
        .flag(false)
        .op(a.getB())
        .strides(B)
        .op(a.getBeta())
        .op(a.getC())
        .strides(C)
        .call(o, "ulmBLAS::gemm" kIndexType);
}

// y := alpha op(A) x + beta y
ULM_LAMK_LOWERING(GemvLowering, lk::GemvOp)
{
    View A = view(rw, o.getA(), o.getTrans() == lk::Trans::trans);
    return Args{rw}
        .dim(A.rows)
        .dim(A.cols)
        .op(a.getAlpha())
        .op(a.getA())
        .strides(A)
        .op(a.getX())
        .lit(1)
        .op(a.getBeta())
        .op(a.getY())
        .lit(1)
        .call(o, "ulmBLAS::gemv" kIndexType);
}

// B := alpha op(A)^-1 B (left) or alpha B op(A)^-1 (right)
ULM_LAMK_LOWERING(TrsmLowering, lk::TrsmOp)
{
    bool trans = o.getTrans() == lk::Trans::trans;
    bool left = o.getSide() == lk::Side::left;
    // right: B op(A)^-1 = (op(A)^-T B^T)^T, a left solve on transposed views
    bool t = left ? trans : !trans;
    View A = view(rw, o.getA(), t), B = view(rw, o.getB(), !left);
    bool lower = (o.getUplo() == lk::Uplo::lower) != t;
    return Args{rw}
        .dim(B.rows)
        .dim(B.cols)
        .op(a.getAlpha())
        .flag(false)
        .flag(o.getDiag() == lk::Diag::unit)
        .op(a.getA())
        .strides(A)
        .op(a.getB())
        .strides(B)
        .call(o, lower ? "ulmBLAS::trlsm" kIndexType
                       : "ulmBLAS::trusm" kIndexType);
}

// x := op(A)^-1 x
ULM_LAMK_LOWERING(TrsvLowering, lk::TrsvOp)
{
    bool t = o.getTrans() == lk::Trans::trans;
    View A = view(rw, o.getA(), t);
    bool lower = (o.getUplo() == lk::Uplo::lower) != t;
    return Args{rw}
        .dim(A.rows)
        .flag(o.getDiag() == lk::Diag::unit)
        .op(a.getA())
        .strides(A)
        .op(a.getX())
        .lit(1)
        .call(o, lower ? "ulmBLAS::trlsv" kIndexType
                       : "ulmBLAS::trusv" kIndexType);
}

// Y := alpha X + Y
ULM_LAMK_LOWERING(GeaxpyLowering, lk::GeaxpyOp)
{
    View X = view(rw, o.getX()), Y = view(rw, o.getY());
    return Args{rw}
        .dim(Y.rows)
        .dim(Y.cols)
        .op(a.getAlpha())
        .op(a.getX())
        .strides(X)
        .op(a.getY())
        .strides(Y)
        .call(o, "ulmBLAS::geaxpy" kIndexType);
}

// X := alpha X
ULM_LAMK_LOWERING(GescalLowering, lk::GescalOp)
{
    View X = view(rw, o.getX());
    return Args{rw}
        .dim(X.rows)
        .dim(X.cols)
        .op(a.getAlpha())
        .op(a.getX())
        .strides(X)
        .call(o, "ulmBLAS::gescal" kIndexType);
}

// A = L L^T in place
ULM_LAMK_LOWERING(PotrfLowering, lk::PotrfOp)
{
    View A = view(rw, o.getA());
    return Args{rw}
        .flag(o.getUplo() == lk::Uplo::lower)
        .dim(A.rows)
        .op(a.getA())
        .strides(A)
        .flag(o.getChecked())
        .call(o, "ulm_lam_potrf");
}

// P A = L U in place
ULM_LAMK_LOWERING(GetrfLowering, lk::GetrfOp)
{
    View A = view(rw, o.getA());
    return Args{rw}
        .dim(A.rows)
        .dim(A.cols)
        .op(a.getA())
        .strides(A)
        .op(a.getIpiv())
        .flag(o.getChecked())
        .call(o, "ulm_lam_getrf");
}

// B := op(A)^-1 B with the LU factors
ULM_LAMK_LOWERING(GetrsLowering, lk::GetrsOp)
{
    View A = view(rw, o.getLU()), B = view(rw, o.getB());
    return Args{rw}
        .flag(o.getTrans() == lk::Trans::trans)
        .flag(false)
        .dim(A.rows)
        .dim(B.cols)
        .op(a.getLU())
        .strides(A)
        .op(a.getIpiv())
        .lit(1)
        .op(a.getB())
        .strides(B)
        .call(o, "ulmBLAS::getrs" kIndexType);
}

#undef ULM_LAMK_LOWERING

// f(inputs..., inits...)                 the external function updates the
//                                        inits in place
struct CallLowering : OpConversionPattern<lk::CallOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(lk::CallOp o, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	if (!cast<DestinationStyleOpInterface>(*o).hasPureBufferSemantics())
	    return rw.notifyMatchFailure(o, "not bufferized");
	SmallVector<Value> ops(a.getInputs());
	llvm::append_range(ops, a.getInits());
	rw.replaceOpWithNewOp<emitc::CallOpaqueOp>(o, TypeRange{},
	                                           o.getCallee(), ops);
	return success();
    }
};

// ---- checks of LAPACK info, emitted into the output ----------------------

const char *kPotrf = R"(template <typename T>
static void
ulm_lam_potrf(bool lower, std::ptrdiff_t n, T *A, std::ptrdiff_t incRow,
              std::ptrdiff_t incCol, bool checked)
{
    std::ptrdiff_t info = ulmBLAS::potrf(lower, n, A, incRow, incCol);
    if (checked && info != 0) {
        std::fprintf(stderr, "ulm_lam: matrix is not positive definite (potrf info %td)\n", info);
        std::abort();
    }
})";

/// Weak default of the global workspace error function: a program may
/// define its own (and then the function returns without computing).
const char *kWorkspaceError =
    R"(extern "C" __attribute__((weak)) void
ulm_lam_workspace_error(int64_t required, int64_t given)
{
    std::fprintf(stderr, "ulm_lam: workspace too small: %lld bytes required, %lld given\n",
                 (long long)required, (long long)given);
    std::abort();
})";

const char *kGetrf = R"(template <typename T>
static void
ulm_lam_getrf(std::ptrdiff_t m, std::ptrdiff_t n, T *A, std::ptrdiff_t incRow,
              std::ptrdiff_t incCol, int32_t *piv, bool checked)
{
    std::ptrdiff_t info = ulmBLAS::getrf(m, n, A, incRow, incCol, piv,
                                         std::ptrdiff_t(1));
    if (checked && info != 0) {
        std::fprintf(stderr, "ulm_lam: matrix is singular (getrf info %td)\n", info);
        std::abort();
    }
})";

// ---- the interface comment ------------------------------------------------

/// What the caller must know about each parameter (LAPACK documents it;
/// here it is generated from the parameter attributes):
///
///   /* gls: matrices by reference (row-major), owned by the caller
///    *   v1  double[2000][200]   in     read only
///    *   v2  double[2000][2000]  inout  may be overwritten (unspecified on
///    exit)
///    *   ...
///    */
std::string
describe(func::FuncOp f)
{
    auto cType = [](Type t) -> std::string {
	if (t.isF64())
	    return "double";
	if (t.isF32())
	    return "float";
	if (t.isIndex())
	    return "size_t";
	if (auto i = dyn_cast<IntegerType>(t))
	    return i.getWidth() == 1
	               ? "bool"
		       : "int" + std::to_string(i.getWidth()) + "_t";
	return "?";
    };
    std::vector<std::array<std::string, 4>> rows;
    unsigned n = f.getNumArguments();
    for (unsigned i = 0; i < n; ++i) {
	Type t = f.getArgumentTypes()[i];
	std::string name = "v" + std::to_string(i + 1), type, intent, what;
	auto has = [&](StringRef a) {
	    return static_cast<bool>(f.getArgAttr(i, a));
	};
	if (auto m = dyn_cast<MemRefType>(t)) {
	    type = cType(m.getElementType());
	    if (has("ulm_lam.workspace")) {
		bool work = i + 2 == n;
		type += " *";
		intent = work ? "work" : "info";
		what = work ? "workspace (bytes)"
		            : "-1: query, 0: allocate, >0: size of work";
	    } else {
		auto dims = dyn_cast_if_present<ArrayAttr>(
		    f.getArgAttr(i, ulm_lam::kDimsAttr));
		unsigned k = 0;
		for (int64_t d : m.getShape()) {
		    std::string s = "?";
		    if (!ShapedType::isDynamic(d))
			s = std::to_string(d);
		    else if (dims && k < dims.size())
			s = "v" +
			    std::to_string(
			        cast<IntegerAttr>(dims[k++]).getInt() + 1);
		    type += "[" + s + "]";
		}
		if (has("bufferize.result")) {
		    intent = "out";
		    what = "written";
		} else if (has("ulm_lam.inout") && has("ulm_lam.updated")) {
		    intent = "inout";
		    what = "updated in place";
		} else if (has("ulm_lam.inout")) {
		    intent = "inout";
		    what = "may be overwritten (unspecified on exit)";
		} else {
		    intent = "in";
		    what = "read only";
		}
	    }
	} else {
	    type = cType(t);
	    intent = "in";
	}
	rows.push_back({name, type, intent, what});
    }
    size_t w0 = 0, w1 = 0, w2 = 0;
    for (auto &r : rows) {
	w0 = std::max(w0, r[0].size());
	w1 = std::max(w1, r[1].size());
	w2 = std::max(w2, r[2].size());
    }
    auto pad = [](const std::string &x, size_t w) {
	return x + std::string(w - x.size(), ' ');
    };
    std::string s = "/* " + f.getSymName().str() +
                    ": matrices by reference (row-major), owned by the caller";
    for (auto &r : rows) {
	std::string line = " *   " + pad(r[0], w0) + "  " + pad(r[1], w1) +
	                   "  " +
	                   (r[3].empty() ? r[2] : pad(r[2], w2) + "  " + r[3]);
	while (!line.empty() && line.back() == ' ')
	    line.pop_back();
	s += "\n" + line;
    }
    return s + "\n */";
}

// ---- the pass -----------------------------------------------------------

struct ConvertUlmLamkToUlmBLAS
    : PassWrapper<ConvertUlmLamkToUlmBLAS, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertUlmLamkToUlmBLAS)

    StringRef
    getArgument() const final
    {
	return "convert-ulm-lamk-to-ulmblas";
    }
    StringRef
    getDescription() const final
    {
	return "Lower bufferized ulm_lamk kernels to EmitC calling the ulmBLAS "
	       "C++ "
	       "interface";
    }
    void
    getDependentDialects(DialectRegistry &registry) const override
    {
	registry.insert<emitc::EmitCDialect>();
    }

    void
    runOnOperation() override
    {
	ModuleOp module = getOperation();
	MLIRContext *ctx = &getContext();
	// sizes as function parameters, not queries of buffers
	ulm_lam::resolveDimOps(module);
	DimTable dims;
	collectDims(module, dims);
	dimTable = &dims;
	llvm::scope_exit reset([] { dimTable = nullptr; });

	// The by-reference convention: no function returns a matrix.
	bool returnsMatrix = false;
	module.walk([&](func::FuncOp f) {
	    for (Type t : f.getResultTypes())
		if (isa<ShapedType>(t)) {
		    f.emitError("convert-ulm-lamk-to-ulmblas: '")
		        << f.getSymName()
		        << "' returns a matrix; matrices are passed by "
		           "reference (the caller allocates and owns them), "
		           "run "
		           "ulm-lam-results-to-out-params and "
		           "buffer-results-to-out-params first";
		    returnsMatrix = true;
		}
	});
	if (returnsMatrix)
	    return signalPassFailure();

	// what the caller must know about the parameters (from their
	// attributes, before the conversion drops them)
	std::vector<std::pair<std::string, std::string>> interfaces;
	for (auto f : module.getOps<func::FuncOp>())
	    if (!f.isExternal() && !f.isPrivate())
		interfaces.push_back({f.getSymName().str(), describe(f)});

	TypeConverter tc;
	tc.addConversion([](Type t) -> std::optional<Type> {
	    if (emitc::isSupportedEmitCType(t))
		return t;
	    return std::nullopt;
	});
	tc.addConversion([](MemRefType t) -> std::optional<Type> {
	    if (!isCArray(t))
		return std::nullopt;
	    return emitc::PointerType::get(t.getElementType());
	});
	populateEmitCSizeTTypeConversions(tc);

	// Which ulmBLAS LAPACK headers and info checks the output needs.
	bool potrf = false, getrf = false, getrs = false;
	bool workspace =
	    static_cast<bool>(module.lookupSymbol("ulm_lam_workspace_error"));
	module.walk([&](Operation *op) {
	    potrf |= isa<lk::PotrfOp>(op);
	    getrf |= isa<lk::GetrfOp>(op);
	    getrs |= isa<lk::GetrsOp>(op);
	});

	RewritePatternSet patterns(ctx);
	patterns.add<AllocLowering, DeallocLowering, CopyLowering, ViewLowering,
	             CastLowering, LoadLowering, StoreLowering,
	             IndexAndLowering, GemmLowering, GemvLowering, TrsmLowering,
	             TrsvLowering, GeaxpyLowering, GescalLowering,
	             PotrfLowering, GetrfLowering, GetrsLowering, CallLowering>(
	    tc, ctx);
	populateArithToEmitCPatterns(tc, patterns);
	populateSCFToEmitCConversionPatterns(patterns, tc);
	populateFuncToEmitCPatterns(tc, patterns, /*lowerToCpp=*/true);

	ConversionTarget conv(*ctx);
	conv.addLegalDialect<emitc::EmitCDialect>();
	conv.addIllegalDialect<memref::MemRefDialect, lk::UlmLamkDialect,
	                       ulm_lam::UlmLamDialect, func::FuncDialect,
	                       scf::SCFDialect, arith::ArithDialect>();
	if (failed(applyPartialConversion(module, conv, std::move(patterns))))
	    return signalPassFailure();

	// Casts left without use (e.g. the sizes of workspace parts, which the
	// pointer arithmetic does not need) would be unused variables in C++.
	for (bool erased = true; erased;) {
	    erased = false;
	    module.walk([&](emitc::CastOp c) {
		if (c->use_empty()) {
		    c->erase();
		    erased = true;
		}
	    });
	}

	// The interface of every public function, for the caller.
	for (auto [name, text] : interfaces)
	    if (auto ef = module.lookupSymbol<emitc::FuncOp>(name)) {
		OpBuilder cb(ef);
		emitc::VerbatimOp::create(cb, ef.getLoc(), text);
	    }

	// C linkage: C callers, C callees.
	module.walk([&](emitc::FuncOp f) {
	    f.setSpecifiersAttr(
	        ArrayAttr::get(ctx, {StringAttr::get(ctx, "extern \"C\"")}));
	});

	// Headers and info checks, in front of everything else.
	OpBuilder b(ctx);
	b.setInsertionPointToStart(module.getBody());
	Location loc = module.getLoc();
	auto include = [&](StringRef h) {
	    emitc::IncludeOp::create(b, loc, h, /*is_standard=*/true);
	};
	for (StringRef h : {"cstddef", "cstdint", "cstdio", "cstdlib",
	                    "cstring", "ulmblas/ulmblas.h"})
	    include(h);
	if (potrf)
	    include("ulmblas/lapack/potrf.h");
	if (getrf)
	    include("ulmblas/lapack/getrf.h");
	if (getrs)
	    include("ulmblas/lapack/getrs.h");
	if (potrf)
	    emitc::VerbatimOp::create(b, loc, kPotrf);
	if (getrf)
	    emitc::VerbatimOp::create(b, loc, kGetrf);
	if (workspace)
	    emitc::VerbatimOp::create(b, loc, kWorkspaceError);
    }
};

} // namespace

namespace ulm_lam {
void
registerConvertUlmLamkToUlmBLASPass()
{
    PassRegistration<ConvertUlmLamkToUlmBLAS>();
}
} // namespace ulm_lam
