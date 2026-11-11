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

#include <array>
#include <string>
#include <vector>

using namespace mlir;
namespace lk = mlir::ulm_lamk;

namespace {

constexpr int64_t kAlignment = 64;

/// Static, identity layout, f32/f64 (rank 1/2) or i32 pivots (rank 1).
bool
isCArray(MemRefType t)
{
    if (!t || !t.getLayout().isIdentity())
	return false;
    Type e = t.getElementType();
    // the workspace: a byte buffer of any size
    if (e.isInteger(8))
	return t.getRank() == 1;
    if (!t.hasStaticShape())
	return false;
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

// ---- memory -------------------------------------------------------------

struct AllocLowering : OpConversionPattern<memref::AllocOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
	MemRefType t = op.getType();
	if (!isCArray(t))
	    return rewriter.notifyMatchFailure(op, "not a static C array");
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

// ---- the workspace ------------------------------------------------------

/// A temporary in the workspace: (T *)(work + offset).
struct ViewLowering : OpConversionPattern<memref::ViewOp>
{
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(memref::ViewOp op, OpAdaptor a,
                    ConversionPatternRewriter &rw) const override
    {
	// (a dynamic size only for byte buffers: the part of a callee)
	if (!isCArray(op.getType()) ||
	    (!op.getSizes().empty() &&
	     !op.getType().getElementType().isInteger(8)))
	    return rw.notifyMatchFailure(op, "not a static C array view");
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
	if (!isCArray(t) || t != op.getTarget().getType())
	    return rewriter.notifyMatchFailure(op, "not a static C array copy");
	rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
	    op, TypeRange{}, "memcpy",
	    ValueRange{adaptor.getTarget(), adaptor.getSource()},
	    rewriter.getArrayAttr({rewriter.getIndexAttr(0),
	                           rewriter.getIndexAttr(1),
	                           rewriter.getI64IntegerAttr(byteSize(t))}));
	return success();
    }
};

// ---- kernels ------------------------------------------------------------

/// op(M) of a row-major matrix (or n x 1 vector) operand in ulmBLAS terms.
struct View
{
    int64_t rows, cols;     // of op(M)
    int64_t incRow, incCol; // strides of op(M)
};

View
view(Value v, bool trans = false)
{
    auto t = cast<MemRefType>(v.getType());
    int64_t r = t.getDimSize(0), c = t.getRank() == 2 ? t.getDimSize(1) : 1;
    if (trans)
	return {c, r, 1, c};
    return {r, c, c, 1};
}

/// Argument builders and the call.
template <typename OpT> struct KernelLowering : OpConversionPattern<OpT>
{
    using OpConversionPattern<OpT>::OpConversionPattern;
    using Rewriter = ConversionPatternRewriter;

    static Attribute
    lit(Rewriter &rw, int64_t v)
    {
	return rw.getI32IntegerAttr(v);
    }
    static Attribute
    arg(Rewriter &rw, int64_t i)
    {
	return rw.getIndexAttr(i);
    }
    static Attribute
    flag(Rewriter &rw, bool b)
    {
	return emitc::OpaqueAttr::get(rw.getContext(), b ? "true" : "false");
    }

    /// All shaped operands must be static row-major C arrays.
    static bool
    allCArrays(Operation *o)
    {
	return llvm::all_of(o->getOperands(), [](Value v) {
	    return !isa<ShapedType>(v.getType()) ||
	           isCArray(dyn_cast<MemRefType>(v.getType()));
	});
    }

    static LogicalResult
    call(Rewriter &rw, Operation *o, StringRef callee, ArrayRef<Value> operands,
         ArrayRef<Attribute> args)
    {
	rw.replaceOpWithNewOp<emitc::CallOpaqueOp>(o, TypeRange{}, callee,
	                                           ValueRange(operands),
	                                           rw.getArrayAttr(args));
	return success();
    }
};

#define ULM_LAMK_LOWERING(NAME, OP)                                            \
    struct NAME : KernelLowering<OP>                                           \
    {                                                                          \
	using KernelLowering::KernelLowering;                                  \
	LogicalResult                                                          \
	matchAndRewrite(OP o, OpAdaptor a, Rewriter &rw) const override        \
	{                                                                      \
	    if (!allCArrays(o))                                                \
		return rw.notifyMatchFailure(                                  \
		    o, "not on static row-major arrays");                      \
	    return lower(o, a, rw);                                            \
	}                                                                      \
	static LogicalResult                                                   \
	lower(OP o, OpAdaptor a, Rewriter &rw);                                \
    };                                                                         \
    LogicalResult NAME::lower(OP o, OpAdaptor a, Rewriter &rw)

// C := alpha op(A) op(B) + beta C        operands: alpha A B beta C
ULM_LAMK_LOWERING(GemmLowering, lk::GemmOp)
{
    View A = view(o.getA(), o.getTransA() == lk::Trans::trans);
    View B = view(o.getB(), o.getTransB() == lk::Trans::trans);
    View C = view(o.getC());
    SmallVector<Value> ops{a.getAlpha(), a.getA(), a.getB(), a.getBeta(),
                           a.getC()};
    return call(rw, o, "ulmBLAS::gemm", ops,
                {lit(rw, C.rows), lit(rw, C.cols), lit(rw, A.cols), arg(rw, 0),
                 flag(rw, false), arg(rw, 1), lit(rw, A.incRow),
                 lit(rw, A.incCol), flag(rw, false), arg(rw, 2),
                 lit(rw, B.incRow), lit(rw, B.incCol), arg(rw, 3), arg(rw, 4),
                 lit(rw, C.incRow), lit(rw, C.incCol)});
}

// y := alpha op(A) x + beta y            operands: alpha A x beta y
ULM_LAMK_LOWERING(GemvLowering, lk::GemvOp)
{
    View A = view(o.getA(), o.getTrans() == lk::Trans::trans);
    SmallVector<Value> ops{a.getAlpha(), a.getA(), a.getX(), a.getBeta(),
                           a.getY()};
    return call(rw, o, "ulmBLAS::gemv", ops,
                {lit(rw, A.rows), lit(rw, A.cols), arg(rw, 0), arg(rw, 1),
                 lit(rw, A.incRow), lit(rw, A.incCol), arg(rw, 2), lit(rw, 1),
                 arg(rw, 3), arg(rw, 4), lit(rw, 1)});
}

// B := alpha op(A)^-1 B (left) or alpha B op(A)^-1 (right)
//                                        operands: alpha A B
ULM_LAMK_LOWERING(TrsmLowering, lk::TrsmOp)
{
    bool trans = o.getTrans() == lk::Trans::trans;
    bool left = o.getSide() == lk::Side::left;
    // right: B op(A)^-1 = (op(A)^-T B^T)^T, a left solve on transposed views
    bool t = left ? trans : !trans;
    View A = view(o.getA(), t), B = view(o.getB(), !left);
    bool lower = (o.getUplo() == lk::Uplo::lower) != t;
    SmallVector<Value> ops{a.getAlpha(), a.getA(), a.getB()};
    return call(rw, o, lower ? "ulmBLAS::trlsm" : "ulmBLAS::trusm", ops,
                {lit(rw, B.rows), lit(rw, B.cols), arg(rw, 0), flag(rw, false),
                 flag(rw, o.getDiag() == lk::Diag::unit), arg(rw, 1),
                 lit(rw, A.incRow), lit(rw, A.incCol), arg(rw, 2),
                 lit(rw, B.incRow), lit(rw, B.incCol)});
}

// x := op(A)^-1 x                        operands: A x
ULM_LAMK_LOWERING(TrsvLowering, lk::TrsvOp)
{
    bool t = o.getTrans() == lk::Trans::trans;
    View A = view(o.getA(), t);
    bool lower = (o.getUplo() == lk::Uplo::lower) != t;
    SmallVector<Value> ops{a.getA(), a.getX()};
    return call(rw, o, lower ? "ulmBLAS::trlsv" : "ulmBLAS::trusv", ops,
                {lit(rw, A.rows), flag(rw, o.getDiag() == lk::Diag::unit),
                 arg(rw, 0), lit(rw, A.incRow), lit(rw, A.incCol), arg(rw, 1),
                 lit(rw, 1)});
}

// Y := alpha X + Y                       operands: alpha X Y
ULM_LAMK_LOWERING(GeaxpyLowering, lk::GeaxpyOp)
{
    View X = view(o.getX()), Y = view(o.getY());
    SmallVector<Value> ops{a.getAlpha(), a.getX(), a.getY()};
    return call(rw, o, "ulmBLAS::geaxpy", ops,
                {lit(rw, Y.rows), lit(rw, Y.cols), arg(rw, 0), arg(rw, 1),
                 lit(rw, X.incRow), lit(rw, X.incCol), arg(rw, 2),
                 lit(rw, Y.incRow), lit(rw, Y.incCol)});
}

// X := alpha X                           operands: alpha X
ULM_LAMK_LOWERING(GescalLowering, lk::GescalOp)
{
    View X = view(o.getX());
    SmallVector<Value> ops{a.getAlpha(), a.getX()};
    return call(rw, o, "ulmBLAS::gescal", ops,
                {lit(rw, X.rows), lit(rw, X.cols), arg(rw, 0), arg(rw, 1),
                 lit(rw, X.incRow), lit(rw, X.incCol)});
}

// A = L L^T in place                     operands: A
ULM_LAMK_LOWERING(PotrfLowering, lk::PotrfOp)
{
    View A = view(o.getA());
    SmallVector<Value> ops{a.getA()};
    return call(rw, o, "ulm_lam_potrf", ops,
                {flag(rw, o.getUplo() == lk::Uplo::lower), lit(rw, A.rows),
                 arg(rw, 0), lit(rw, A.incRow), lit(rw, A.incCol),
                 flag(rw, o.getChecked())});
}

// P A = L U in place                     operands: A ipiv
ULM_LAMK_LOWERING(GetrfLowering, lk::GetrfOp)
{
    View A = view(o.getA());
    SmallVector<Value> ops{a.getA(), a.getIpiv()};
    return call(rw, o, "ulm_lam_getrf", ops,
                {lit(rw, A.rows), lit(rw, A.cols), arg(rw, 0),
                 lit(rw, A.incRow), lit(rw, A.incCol), arg(rw, 1),
                 flag(rw, o.getChecked())});
}

// B := op(A)^-1 B with the LU factors    operands: LU ipiv B
ULM_LAMK_LOWERING(GetrsLowering, lk::GetrsOp)
{
    View A = view(o.getLU()), B = view(o.getB());
    SmallVector<Value> ops{a.getLU(), a.getIpiv(), a.getB()};
    return call(rw, o, "ulmBLAS::getrs", ops,
                {flag(rw, o.getTrans() == lk::Trans::trans), flag(rw, false),
                 lit(rw, A.rows), lit(rw, B.cols), arg(rw, 0),
                 lit(rw, A.incRow), lit(rw, A.incCol), arg(rw, 1), lit(rw, 1),
                 arg(rw, 2), lit(rw, B.incRow), lit(rw, B.incCol)});
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
ulm_lam_potrf(bool lower, int n, T *A, int incRow, int incCol, bool checked)
{
    int info = ulmBLAS::potrf(lower, n, A, incRow, incCol);
    if (checked && info != 0) {
        std::fprintf(stderr, "ulm_lam: matrix is not positive definite (potrf info %d)\n", info);
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
ulm_lam_getrf(int m, int n, T *A, int incRow, int incCol, int32_t *piv, bool checked)
{
    int info = ulmBLAS::getrf(m, n, A, incRow, incCol, piv, 1);
    if (checked && info != 0) {
        std::fprintf(stderr, "ulm_lam: matrix is singular (getrf info %d)\n", info);
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
		if (m.hasStaticShape())
		    for (int64_t d : m.getShape())
			type += "[" + std::to_string(d) + "]";
		else
		    type += " *";
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
	             CastLowering, LoadLowering, StoreLowering, GemmLowering,
	             GemvLowering, TrsmLowering, TrsvLowering, GeaxpyLowering,
	             GescalLowering, PotrfLowering, GetrfLowering,
	             GetrsLowering, CallLowering>(tc, ctx);
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
