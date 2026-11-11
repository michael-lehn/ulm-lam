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
//     Such functions must not have callers inside the module.
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
isStaticTensor(Type t)
{
    auto rt = dyn_cast<RankedTensorType>(t);
    return rt && rt.hasStaticShape() && !rt.getEncoding();
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
	Value v = result;
	OpOperand *first = nullptr;
	while (auto dps = v.getDefiningOp<DestinationStyleOpInterface>()) {
	    OpOperand *init = dps.getTiedOpOperand(cast<OpResult>(v));
	    if (!init)
		break;
	    first = init;
	    v = init->get();
	}
	if (!first || v.getDefiningOp<tensor::EmptyOp>())
	    return;
	Operation *op = first->getOwner();
	if (isa<ulm_lamk::CallOp>(op))
	    return; // an external function updates its own argument
	OpBuilder b(op);
	Location loc = op->getLoc();
	auto t = bufferization::ToTensorOp::create(b, loc, v.getType(), out);
	t.setRestrict(true);
	t.setWritable(true);
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
	    if (auto uses = SymbolTable::getSymbolUses(f, module);
	        !uses || !uses->empty())
		continue; // called in the module: buffer-results-to-out-params
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
	llvm::StringSet<> outParams, inPlace;
	for (auto f : module.getOps<func::FuncOp>()) {
	    if (!f.isExternal() || f.getNumResults() == 0)
		continue;
	    if (!llvm::all_of(f.getResultTypes(), isStaticTensor))
		continue; // dynamic shapes: needs a size query, not handled yet
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
	    if (failed(f.insertArguments(idx, types, attrs, locs)) ||
	        failed(f.eraseResults(llvm::BitVector(r, true))))
		return f.emitError("cannot rewrite signature");
	    outParams.insert(f.getSymName());
	}

	SmallVector<func::CallOp> calls;
	module.walk([&](func::CallOp c) {
	    if (outParams.contains(c.getCallee()) ||
	        inPlace.contains(c.getCallee()))
		calls.push_back(c);
	});
	for (func::CallOp call : calls) {
	    OpBuilder b(call);
	    Location loc = call.getLoc();
	    if (inPlace.contains(call.getCallee())) {
		// X' = f(..., X)  ->  X' = ulm_lamk.call @f(...) outs(X)
		unsigned r = call.getNumResults(), n = call.getNumOperands();
		SmallVector<Value> inputs(call.getOperands().take_front(n - r));
		SmallVector<Value> inits;
		for (Value v : call.getOperands().drop_front(n - r)) {
		    // still needed after the call: update a copy
		    if (!v.hasOneUse()) {
			Value e = tensor::EmptyOp::create(
			    b, loc,
			    cast<RankedTensorType>(v.getType()).getShape(),
			    cast<RankedTensorType>(v.getType())
			        .getElementType());
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
		continue;
	    }
	    SmallVector<Value> operands(call.getOperands());
	    SmallVector<Value> buffers;
	    for (Type t : call.getResultTypes()) {
		auto alloc = memref::AllocOp::create(b, loc, bufferTypeFor(t));
		alloc.setAlignment(64);
		buffers.push_back(alloc);
		operands.push_back(alloc);
	    }
	    func::CallOp::create(b, loc, call.getCallee(), TypeRange{},
	                         operands);
	    for (auto [res, buf] : llvm::zip(call.getResults(), buffers)) {
		auto t = bufferization::ToTensorOp::create(b, loc,
		                                           res.getType(), buf);
		t.setRestrict(true);
		t.setWritable(true);
		res.replaceAllUsesWith(t.getResult());
	    }
	    call.erase();
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
		Type at = f.getArgumentTypes()[i], rt = f.getResultTypes()[j];
		if (!isStaticTensor(at) || at != rt)
		    return f.emitError()
		           << "result " << j << " and argument " << i
		           << " must have the same static tensor type";
		inout.push_back({j, i});
	    }
	    if (inout.empty())
		continue;
	    if (auto uses = SymbolTable::getSymbolUses(f, module);
	        uses && !uses->empty())
		return f.emitError("functions with ")
		       << kInOutAttr
		       << " results must not be called inside the module";

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
