//===- UlmLamResultsToOutParams.cpp - by-reference matrix results ---------===//
//
// ulm-lam-results-to-out-params: rewrite matrix-valued results into
// destination-passing style *before* bufferization, so that every matrix
// is passed by reference and One-Shot Bufferize plus buffer-loop-hoisting
// can place all buffers.
//
//  1. External declarations returning statically shaped tensors
//
//       func.func private @foo(index) -> tensor<500x500xf64>
//       %B = func.call @foo(%i) : (index) -> tensor<500x500xf64>
//
//     become
//
//       func.func private @foo(index, memref<500x500xf64>)
//       %Bm = memref.alloc() : memref<500x500xf64>
//       func.call @foo(%i, %Bm) : (index, memref<500x500xf64>) -> ()
//       %B = bufferization.to_tensor %Bm restrict writable : ...
//
//     i.e. X = foo(i) is rewritten into foo(i, X) with X by reference.
//     An external function whose results are marked {ulm_lam.inout = k}
//     updates its trailing arguments in place instead:
//
//       func.func private @update(index, tensor<..>) -> (tensor<..>
//       {ulm_lam.inout = 1}) %R1 = func.call @update(%i, %R0)
//
//     becomes `func.func private @update(index, memref<..>)` and
//     `%R1 = ulm_lamk.call @update(%i) outs(%R0)`: in destination-passing
//     style, so bufferization keeps R in one buffer, also in a loop.
//
//     The C ABI is `void foo(size_t i, double *X)` (EmitC) or the memref
//     descriptor equivalent (LLVM).  The allocation sits where the call
//     was; buffer-loop-hoisting moves it out of loops.
//
//  2. Defined functions whose tensor result j carries `{ulm_lam.inout = i}`
//     (result j is the updated value of argument i, e.g. X += A*B*C)
//
//       func.func @f(..., %X: tensor<..>, ...) -> (tensor<..> {ulm_lam.inout =
//       2})
//
//     become
//
//       func.func @f(..., %X: memref<..>, ...)
//       %X0 = bufferization.to_tensor %X restrict writable
//       ...
//       bufferization.materialize_in_destination %Xn in restrict writable %X
//
//     so X is updated in place through the reference and never copied.
//     A call of such a function passes X by reference (a copy where the
//     caller must not overwrite it) and takes the buffer after the call as
//     the new value: a chain of calls updates one buffer (LAPACK style).
//
//  3. The other tensor results of public functions without callers in the
//     module become out parameters (appended, attribute bufferize.result):
//
//       func.func @f(...) -> (tensor<?x1xf64> {ulm_lam.dims = [1]})
//
//     becomes
//
//       func.func @f(..., %b: memref<?x1xf64> {ulm_lam.dims = [1],
//                                               bufferize.result})
//       bufferization.materialize_in_destination %r in restrict writable %b
//
//     and eliminate-empty-tensors lets the kernel that computes the result
//     write into %b directly (also for dynamic shapes, which
//     buffer-results-to-out-params{hoist-static-allocs} cannot do).
//
//  4. Tensor arguments of public functions (passed by value) are marked
//     bufferization.writable = false: the caller's matrix is never
//     overwritten, bufferization copies where a kernel works in place.
//
// Registered by lib/Plugin.cpp (mlir-opt --load-pass-plugin=...).
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kInOutAttr = "ulm_lam.inout";
constexpr llvm::StringLiteral kInAttr = "ulm_lam.in";
constexpr llvm::StringLiteral kUpdatedAttr = "ulm_lam.updated";

MemRefType
bufferTypeFor(Type t)
{
    auto rt = cast<RankedTensorType>(t);
    return MemRefType::get(rt.getShape(), rt.getElementType());
}

bool
isPlainTensor(Type t)
{
    auto rt = dyn_cast<RankedTensorType>(t);
    return rt && !rt.getEncoding();
}

/// The dims attribute of each result of f (before the signature changes).
SmallVector<Attribute>
resultDims(func::FuncOp f)
{
    SmallVector<Attribute> dims;
    for (unsigned j = 0, e = f.getNumResults(); j < e; ++j)
	dims.push_back(f.getResultAttr(j, ulm_lam::kDimsAttr));
    return dims;
}

/// X = f(...) becomes f(..., X) at every call of f: the results in
/// `results` are allocated by the caller (sizes from the callee's
/// ulm_lam.dims and the call's operands) and passed by reference.
LogicalResult
rewriteCallSites(ModuleOp module, StringRef name, ArrayRef<unsigned> results,
                 ArrayRef<Attribute> dims)
{
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp c) {
	if (c.getCallee() == name)
	    calls.push_back(c);
    });
    for (func::CallOp call : calls) {
	OpBuilder b(call);
	Location loc = call.getLoc();
	SmallVector<Value> operands(call.getOperands());
	SmallVector<Value> buffers;
	for (unsigned j : results) {
	    auto t = cast<RankedTensorType>(call.getResult(j).getType());
	    SmallVector<Value> sizes;
	    auto d = dyn_cast_if_present<ArrayAttr>(dims[j]);
	    unsigned k = 0;
	    for (int64_t s : t.getShape()) {
		if (!ShapedType::isDynamic(s))
		    continue;
		if (!d || k >= d.size())
		    return call.emitError("ulm-lam-results-to-out-params: the "
		                          "size of a result of @")
		           << name << " is unknown (missing "
		           << ulm_lam::kDimsAttr << ")";
		sizes.push_back(operands[cast<IntegerAttr>(d[k++]).getInt()]);
	    }
	    auto alloc =
	        memref::AllocOp::create(b, loc, bufferTypeFor(t), sizes);
	    alloc.setAlignment(64);
	    buffers.push_back(alloc);
	}
	llvm::append_range(operands, buffers);
	SmallVector<Type> kept;
	SmallVector<unsigned> keptIdx;
	for (unsigned j = 0, e = call.getNumResults(); j < e; ++j)
	    if (!llvm::is_contained(results, j)) {
		kept.push_back(call.getResult(j).getType());
		keptIdx.push_back(j);
	    }
	auto n = func::CallOp::create(b, loc, name, kept, operands);
	for (auto [j, r] : llvm::zip(keptIdx, n.getResults()))
	    call.getResult(j).replaceAllUsesWith(r);
	for (auto [j, buf] : llvm::zip(results, buffers)) {
	    auto t = bufferization::ToTensorOp::create(
	        b, loc, call.getResult(j).getType(), buf);
	    t.setRestrict(true);
	    t.setWritable(true);
	    call.getResult(j).replaceAllUsesWith(t.getResult());
	}
	call.erase();
    }
    return success();
}

struct UlmLamResultsToOutParams
    : PassWrapper<UlmLamResultsToOutParams, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UlmLamResultsToOutParams)

    StringRef
    getArgument() const final
    {
	return "ulm-lam-results-to-out-params";
    }
    StringRef
    getDescription() const final
    {
	return "Pass matrix results by reference: X = foo(i) becomes foo(i, "
	       "X); "
	       "results marked {ulm_lam.inout = i} update argument i in place";
    }
    void
    getDependentDialects(DialectRegistry &registry) const override
    {
	registry.insert<memref::MemRefDialect, ulm_lamk::UlmLamkDialect,
	                tensor::TensorDialect,
	                bufferization::BufferizationDialect>();
    }

    void
    runOnOperation() override
    {
	ModuleOp module = getOperation();
	if (failed(rewriteDeclarations(module)) ||
	    failed(rewriteInOut(module)) || failed(rewriteResults(module)))
	    return signalPassFailure();
	if (failed(markIntents(module)))
	    return signalPassFailure();
    }

    // -- 3. results of public functions --------------------------------------
    /// The kernels that compute `result` update their destination in place
    /// (a chain of DPS ops).  If the chain starts from a matrix other than a
    /// new one (tensor.empty, which eliminate-empty-tensors replaces by the
    /// out parameter), e.g. from a copy of a read-only argument B in
    /// X = A^-1 B, the chain starts from a copy in the out parameter
    /// instead: the solve then works in `out` and the final
    /// materialize_in_destination is a copy to itself, which
    /// canonicalisation removes.
    static void
    startInDestination(Value result, Value out)
    {
	// a choice between plans: every branch computes into `out`
	if (auto ifOp = result.getDefiningOp<scf::IfOp>()) {
	    unsigned r = cast<OpResult>(result).getResultNumber();
	    for (Block *blk : {ifOp.thenBlock(), ifOp.elseBlock()})
		if (blk)
		    startInDestination(blk->getTerminator()->getOperand(r),
		                       out);
	    return;
	}
	// the new value of an argument a callee updated, which got a copy:
	// the copy goes into `out` instead, and the callee works there
	// (x = copy(y); foo!(..., x) for a MATLAB-style wrapper of foo!)
	if (auto tt = result.getDefiningOp<bufferization::ToTensorOp>())
	    if (auto tb =
	            tt.getBuffer().getDefiningOp<bufferization::ToBufferOp>())
		if (auto mat =
		        tb.getTensor()
		            .getDefiningOp<
		                bufferization::MaterializeInDestinationOp>())
		    if (mat.getDest().getDefiningOp<tensor::EmptyOp>() &&
		        !isa<scf::IfOp>(
		            result.getParentBlock()->getParentOp())) {
			OpBuilder b(mat);
			auto t = bufferization::ToTensorOp::create(
			    b, mat.getLoc(), mat.getDest().getType(), out);
			t.setRestrict(true);
			t.setWritable(true);
			mat.getDestMutable().set(t.getResult());
			return;
		    }
	Value v = result;
	OpOperand *first = nullptr;
	while (auto dps = v.getDefiningOp<DestinationStyleOpInterface>()) {
	    OpOperand *init = dps.getTiedOpOperand(cast<OpResult>(v));
	    if (!init)
		break;
	    first = init;
	    v = init->get();
	}
	if (!first)
	    return;
	Operation *op = first->getOwner();
	if (v.getDefiningOp<tensor::EmptyOp>() &&
	    !isa<scf::IfOp>(result.getParentBlock()->getParentOp())) {
	    // outside of a branch, eliminate-empty-tensors does it
	    return;
	}
	if (isa<ulm_lamk::CallOp>(op))
	    return; // an external function updates its own argument
	OpBuilder b(op);
	Location loc = op->getLoc();
	auto t = bufferization::ToTensorOp::create(b, loc, v.getType(), out);
	t.setRestrict(true);
	t.setWritable(true);
	if (v.getDefiningOp<tensor::EmptyOp>()) {
	    first->set(t.getResult()); // a new matrix: the out buffer itself
	    return;
	}
	Value c = bufferization::MaterializeInDestinationOp::create(
	              b, loc, v.getType(), v, t.getResult())
	              .getResult();
	first->set(c);
    }

    static LogicalResult
    rewriteResults(ModuleOp module)
    {
	MLIRContext *ctx = module.getContext();
	for (auto f : module.getOps<func::FuncOp>()) {
	    if (f.isExternal() || f.isPrivate() || f.getNumResults() == 0)
		continue;
	    SmallVector<Attribute> dims = resultDims(f);
	    SmallVector<unsigned> results;
	    for (unsigned j = 0, e = f.getNumResults(); j < e; ++j)
		if (isa<RankedTensorType>(f.getResultTypes()[j]))
		    results.push_back(j);
	    if (results.empty())
		continue;
	    unsigned n = f.getNumArguments();
	    SmallVector<unsigned> idx(results.size(), n);
	    SmallVector<Type> types;
	    SmallVector<DictionaryAttr> attrs;
	    SmallVector<Location> locs(results.size(), f.getLoc());
	    for (unsigned j : results) {
		types.push_back(bufferTypeFor(f.getResultTypes()[j]));
		NamedAttrList a;
		a.set("bufferize.result", UnitAttr::get(ctx));
		if (Attribute d = f.getResultAttr(j, ulm_lam::kDimsAttr))
		    a.set(ulm_lam::kDimsAttr, d);
		attrs.push_back(a.getDictionary(ctx));
	    }
	    if (failed(f.insertArguments(idx, types, attrs, locs)))
		return f.emitError("cannot add out parameters");
	    llvm::BitVector drop(f.getNumResults());
	    for (unsigned j : results)
		drop.set(j);
	    SmallVector<func::ReturnOp> returns;
	    f.walk([&](func::ReturnOp r) { returns.push_back(r); });
	    for (func::ReturnOp ret : returns) {
		OpBuilder rb(ret);
		for (auto [k, j] : llvm::enumerate(results)) {
		    Value out = f.getArgument(n + k);
		    startInDestination(ret.getOperand(j), out);
		    auto m = bufferization::MaterializeInDestinationOp::create(
		        rb, ret.getLoc(), TypeRange{}, ret.getOperand(j), out);
		    m.setRestrict(true);
		    m.setWritable(true);
		}
		SmallVector<Value> kept;
		for (unsigned j = 0, e = ret.getNumOperands(); j < e; ++j)
		    if (!drop.test(j))
			kept.push_back(ret.getOperand(j));
		func::ReturnOp::create(rb, ret.getLoc(), kept);
		ret.erase();
	    }
	    if (failed(f.eraseResults(drop)))
		return f.emitError("cannot drop results");
	    // ... and its callers in the module pass the buffers
	    if (failed(rewriteCallSites(module, f.getSymName(), results, dims)))
		return failure();
	}
	return success();
    }

    // -- 4. parameter intents: in, inout, out --------------------------------
    //
    // As in LAPACK, a routine may overwrite the matrices it receives; here
    // the parameter list says so.  A matrix argument of a public function
    // is
    //   {ulm_lam.in}     (the default) read only: if a kernel would overwrite
    //                 it, the function works on a copy (in its workspace);
    //   {ulm_lam.inout}  the function may overwrite it and use it as
    //                 workspace; its contents on exit are those of the
    //                 result marked {ulm_lam.inout = k}, or unspecified
    //                 (LAPACK: "on exit, A is destroyed") if there is none.
    // Results are {ulm_lam.out}: buffers the caller provides (out parameters)
    // that the function writes, and also uses while computing them.
    // One-Shot Bufferize treats function arguments as writable by default;
    // {ulm_lam.in} arguments get bufferization.writable = false.
    static LogicalResult
    markIntents(ModuleOp module)
    {
	MLIRContext *ctx = module.getContext();
	for (auto f : module.getOps<func::FuncOp>()) {
	    if (f.isExternal() || f.isPrivate())
		continue;
	    for (unsigned i = 0, e = f.getNumArguments(); i < e; ++i) {
		if (!isa<RankedTensorType>(f.getArgumentTypes()[i]))
		    continue;
		bool in = static_cast<bool>(f.getArgAttr(i, kInAttr));
		bool inout = static_cast<bool>(f.getArgAttr(i, kInOutAttr));
		if (in && inout)
		    return f.emitError() << "argument " << i << " is both "
		                         << kInAttr << " and " << kInOutAttr;
		if (!inout)
		    f.setArgAttr(i, kInAttr, UnitAttr::get(ctx));
		f.setArgAttr(i, "bufferization.writable",
		             BoolAttr::get(ctx, inout));
	    }
	}
	return success();
    }

    // -- 1. external declarations and their call sites ----------------------
    LogicalResult
    rewriteDeclarations(ModuleOp module)
    {
	llvm::StringSet<> inPlace;
	llvm::StringMap<SmallVector<Attribute>> outParams; // result dims
	for (auto f : module.getOps<func::FuncOp>()) {
	    if (!f.isExternal() || f.getNumResults() == 0)
		continue;
	    if (!llvm::all_of(f.getResultTypes(), isPlainTensor))
		continue;
	    unsigned n = f.getNumArguments(), r = f.getNumResults();
	    // results that update an argument in place: {ulm_lam.inout = k}
	    unsigned marked = 0;
	    for (unsigned j = 0; j < r; ++j)
		if (auto a =
		        f.getResultAttrOfType<IntegerAttr>(j, kInOutAttr)) {
		    unsigned k = a.getInt();
		    if (k != n - r + j ||
		        f.getArgumentTypes()[k] != f.getResultTypes()[j])
			return f.emitError()
			       << "external function with " << kInOutAttr
			       << " results: result j must update argument "
			          "n - r + j (the trailing arguments), with "
			          "the "
			          "same type";
		    ++marked;
		}
	    if (marked != 0 && marked != r)
		return f.emitError() << "external function: either all or no "
		                        "results are "
		                     << kInOutAttr;
	    if (marked) {
		// the trailing arguments become references, the results go
		SmallVector<Type> inputs(f.getArgumentTypes());
		for (unsigned k = n - r; k < n; ++k)
		    inputs[k] = bufferTypeFor(inputs[k]);
		if (failed(f.eraseResults(llvm::BitVector(r, true))))
		    return f.emitError("cannot rewrite signature");
		f.setFunctionType(
		    FunctionType::get(f.getContext(), inputs, {}));
		inPlace.insert(f.getSymName());
		continue;
	    }
	    SmallVector<unsigned> idx(r, n);
	    SmallVector<Type> types;
	    for (Type t : f.getResultTypes())
		types.push_back(bufferTypeFor(t));
	    SmallVector<DictionaryAttr> attrs(
	        r, DictionaryAttr::get(f.getContext()));
	    SmallVector<Location> locs(r, f.getLoc());
	    SmallVector<Attribute> dims = resultDims(f);
	    if (failed(f.insertArguments(idx, types, attrs, locs)) ||
	        failed(f.eraseResults(llvm::BitVector(r, true))))
		return f.emitError("cannot rewrite signature");
	    outParams[f.getSymName()] = dims;
	}
	for (auto &[name, dims] : outParams) {
	    SmallVector<unsigned> all =
	        llvm::to_vector(llvm::seq<unsigned>(0, dims.size()));
	    if (failed(rewriteCallSites(module, name, all, dims)))
		return failure();
	}

	SmallVector<func::CallOp> calls;
	module.walk([&](func::CallOp c) {
	    if (inPlace.contains(c.getCallee()))
		calls.push_back(c);
	});
	for (func::CallOp call : calls) {
	    OpBuilder b(call);
	    Location loc = call.getLoc();
	    {
		// X' = f(..., X)  ->  X' = ulm_lamk.call @f(...) outs(X)
		unsigned r = call.getNumResults(), n = call.getNumOperands();
		SmallVector<Value> inputs(call.getOperands().take_front(n - r));
		SmallVector<Value> inits;
		for (Value v : call.getOperands().drop_front(n - r)) {
		    // still needed after the call: update a copy
		    if (!v.hasOneUse()) {
			auto vt = cast<RankedTensorType>(v.getType());
			FailureOr<SmallVector<Value>> sizes =
			    ulm_lam::dynamicSizes(b, loc, vt, v, 0, v, 1);
			if (failed(sizes))
			    return call.emitError(
			        "unknown size of an argument");
			Value e = tensor::EmptyOp::create(
			    b, loc, vt.getShape(), vt.getElementType(), *sizes);
			v = bufferization::MaterializeInDestinationOp::create(
			        b, loc, v.getType(), v, e)
			        .getResult();
		    }
		    inits.push_back(v);
		}
		auto c =
		    ulm_lamk::CallOp::create(b, loc, call.getResultTypes(),
		                             call.getCallee(), inputs, inits);
		call.replaceAllUsesWith(c.getResults());
		call.erase();
	    }
	}
	return success();
    }

    // -- 2. in/out results of defined functions ------------------------------
    LogicalResult
    rewriteInOut(ModuleOp module)
    {
	for (auto f : module.getOps<func::FuncOp>()) {
	    if (f.isExternal())
		continue;
	    SmallVector<std::pair<unsigned, unsigned>> inout; // (result, arg)
	    for (unsigned j = 0, e = f.getNumResults(); j < e; ++j) {
		auto a = f.getResultAttrOfType<IntegerAttr>(j, kInOutAttr);
		if (!a)
		    continue;
		unsigned i = a.getInt();
		if (i >= f.getNumArguments())
		    return f.emitError()
		           << kInOutAttr << " = " << i << ": no such argument";
		// (the parameter may carry properties, e.g. spd, its new value
		// not)
		auto at = dyn_cast<RankedTensorType>(f.getArgumentTypes()[i]);
		auto rt = dyn_cast<RankedTensorType>(f.getResultTypes()[j]);
		if (!at || !rt || !isPlainTensor(rt) ||
		    at.getShape() != rt.getShape() ||
		    at.getElementType() != rt.getElementType())
		    return f.emitError()
		           << "result " << j << " and argument " << i
		           << " must have the same tensor type";
		inout.push_back({j, i});
	    }
	    if (inout.empty())
		continue;

	    // Arguments become memrefs, read once through to_tensor.
	    Block &entry = f.getBody().front();
	    OpBuilder b(&entry, entry.begin());
	    SmallVector<Type> inputs(f.getArgumentTypes());
	    for (auto [j, i] : inout) {
		if (f.getArgAttr(i, kInAttr))
		    return f.emitError()
		           << "argument " << i << " is " << kInAttr
		           << " but result " << j << " updates it";
		// the parameter list says it: inout, and holds the result
		f.setArgAttr(i, kInOutAttr, UnitAttr::get(f.getContext()));
		f.setArgAttr(i, kUpdatedAttr, UnitAttr::get(f.getContext()));
		BlockArgument arg = f.getArgument(i);
		Type tensorTy = arg.getType();
		arg.setType(bufferTypeFor(tensorTy));
		inputs[i] = arg.getType();
		auto t = bufferization::ToTensorOp::create(b, f.getLoc(),
		                                           tensorTy, arg);
		t.setRestrict(true);
		t.setWritable(true);
		arg.replaceAllUsesExcept(t.getResult(), t);
	    }

	    // Results are written back in place and dropped.
	    llvm::BitVector drop(f.getNumResults());
	    for (auto [j, i] : inout)
		drop.set(j);
	    SmallVector<func::ReturnOp> returns;
	    f.walk([&](func::ReturnOp r) { returns.push_back(r); });
	    for (func::ReturnOp ret : returns) {
		OpBuilder rb(ret);
		for (auto [j, i] : inout) {
		    auto m = bufferization::MaterializeInDestinationOp::create(
		        rb, ret.getLoc(), TypeRange{}, ret.getOperand(j),
		        f.getArgument(i));
		    m.setRestrict(true);
		    m.setWritable(true);
		}
		SmallVector<Value> kept;
		for (unsigned j = 0, e = ret.getNumOperands(); j < e; ++j)
		    if (!drop.test(j))
			kept.push_back(ret.getOperand(j));
		func::ReturnOp::create(rb, ret.getLoc(), kept);
		ret.erase();
	    }
	    f.setFunctionType(
	        FunctionType::get(f.getContext(), inputs, f.getResultTypes()));
	    if (failed(f.eraseResults(drop)))
		return f.emitError("cannot drop in/out results");
	    if (failed(rewriteUpdatingCalls(module, f, inout)))
		return failure();
	}
	return success();
    }

    /// May the caller let a callee overwrite v?  Not an argument of its own
    /// that is only read ({ulm_lam.in}, or without inout), not a value it
    /// still uses after the call.
    static bool
    overwritable(Value v)
    {
	if (!v.hasOneUse())
	    return false;
	if (auto a = dyn_cast<BlockArgument>(v))
	    if (auto f = dyn_cast<func::FuncOp>(a.getOwner()->getParentOp()))
		return static_cast<bool>(
		    f.getArgAttr(a.getArgNumber(), kInOutAttr));
	return true;
    }

    /// The calls of f, which updates the arguments in `inout` in place:
    ///   %r..., %X1 = call @f(..., %X0, ...)
    /// becomes
    ///   %m = bufferization.to_buffer %X0     (a copy if the caller must not
    ///   call @f(..., %m, ...)                 overwrite X0)
    ///   %X1 = bufferization.to_tensor %m restrict writable
    /// so a chain of such calls (LAPACK style: y updated by every call)
    /// works in one buffer, without copies.
    static LogicalResult
    rewriteUpdatingCalls(ModuleOp module, func::FuncOp f,
                         ArrayRef<std::pair<unsigned, unsigned>> inout)
    {
	SmallVector<func::CallOp> calls;
	module.walk([&](func::CallOp c) {
	    if (c.getCallee() == f.getSymName())
		calls.push_back(c);
	});
	for (func::CallOp call : calls) {
	    OpBuilder b(call);
	    Location loc = call.getLoc();
	    SmallVector<Value> ops(call.getOperands());
	    SmallVector<std::pair<unsigned, Value>> bufs; // (result, buffer)
	    for (auto [j, i] : inout) {
		Value v = ops[i];
		auto vt = cast<RankedTensorType>(v.getType());
		if (!overwritable(v)) {
		    FailureOr<SmallVector<Value>> sizes =
		        ulm_lam::dynamicSizes(b, loc, vt, v, 0, v, 1);
		    if (failed(sizes))
			return call.emitError("unknown size of an argument");
		    Value e = tensor::EmptyOp::create(
		        b, loc, vt.getShape(), vt.getElementType(), *sizes);
		    v = bufferization::MaterializeInDestinationOp::create(
		            b, loc, v.getType(), v, e)
		            .getResult();
		}
		Value m = bufferization::ToBufferOp::create(
		    b, loc, bufferTypeFor(vt), v);
		ops[i] = m;
		bufs.push_back({j, m});
	    }
	    SmallVector<Type> kept;
	    SmallVector<unsigned> keptIdx;
	    for (unsigned j = 0, e = call.getNumResults(); j < e; ++j)
		if (llvm::none_of(bufs,
		                  [&](auto &p) { return p.first == j; })) {
		    kept.push_back(call.getResult(j).getType());
		    keptIdx.push_back(j);
		}
	    auto n = func::CallOp::create(b, loc, call.getCallee(), kept, ops);
	    for (auto [j, r] : llvm::zip(keptIdx, n.getResults()))
		call.getResult(j).replaceAllUsesWith(r);
	    for (auto [j, m] : bufs) {
		auto t = bufferization::ToTensorOp::create(
		    b, loc, call.getResult(j).getType(), m);
		t.setRestrict(true);
		t.setWritable(true);
		call.getResult(j).replaceAllUsesWith(t.getResult());
	    }
	    call.erase();
	}
	return success();
    }
};

} // namespace

namespace ulm_lam {
void
registerResultsToOutParamsPass()
{
    PassRegistration<UlmLamResultsToOutParams>();
}
} // namespace ulm_lam
