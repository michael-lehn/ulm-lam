//===- ulm-lam.cpp - the ulm.lam compiler: ulm_lam IR to C++ or Julia -----===//
//
//   ulm-lam [--backend=ulmblas|cblas] [--emit=cpp|julia]
//           [--select-options=...] [--no-licm] [--selected=FILE]
//           input.mlir -o output
//
// The pipeline of scripts/ulm-lam-compile.sh in one process, without
// mlir-opt, a plugin or mlir-translate: selection of the kernels,
// bufferization, lowering to EmitC, and then either C++ (EmitC's
// translation, as mlir-translate --mlir-to-cpp) or Julia (emitJulia below:
// the same program calling the kernels of UlmLAM.Kernels, so that Julia
// needs no C++ compiler).  The frontends run this tool; linked statically
// against MLIR (Multi-Level Intermediate Representation) it is the one
// binary that ulm.lam needs.
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace ulm_lam {
void
registerResultsToOutParamsPass();
void
registerConvertUlmLamkToUlmBLASPass();
void
registerUlmLamPasses();
void
registerWorkspacePass();
void
registerInfoPass();
} // namespace ulm_lam

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input.mlir>"),
                                      cl::init("-"));
static cl::opt<std::string> outputFile("o", cl::desc("output file"),
                                       cl::init("-"));
static cl::opt<std::string> backend("backend",
                                    cl::desc("ulmblas or cblas (the kernels "
                                             "the generated code calls)"),
                                    cl::init("ulmblas"));
static cl::opt<std::string> emit("emit", cl::desc("cpp or julia"),
                                 cl::init("cpp"));
static cl::opt<std::string>
    selectOptions("select-options",
                  cl::desc("options of ulm-lam-select-kernels"), cl::init(""));
static cl::opt<bool> noLicm("no-licm",
                            cl::desc("no loop-invariant code motion"),
                            cl::init(false));
static cl::opt<std::string>
    selectedFile("selected",
                 cl::desc("write the IR after kernel selection to this file"),
                 cl::init(""));
static cl::opt<std::string> stagesPrefix(
    "stages", cl::desc("write the IR after each stage to PREFIX.<stage>.mlir"),
    cl::init(""));

namespace {

// ---------------------------------------------------------------------------
// Julia: the EmitC program as a Julia program
// ---------------------------------------------------------------------------
//
// Every EmitC operation (before form-expressions: one operation, one
// statement) becomes one assignment to a local variable; C's lvalues are
// local variables (emitc.variable) or pointer elements (emitc.subscript,
// read with unsafe_load and written with unsafe_store!).  All integers are
// Int64 (sizes are below 2^63), pointers are typed (Ptr{Float64}), and
// pointer arithmetic counts elements as in C (Kernels.at).  The kernels of
// the backend (ulm_lam_cblas::gemm and the helpers ulm_lam_copy, ...) are
// the functions of UlmLAM.Kernels, bound to K; external functions are
// called by name (the runtime defines them in the module it evaluates the
// program in); the functions of the program are lam_<name>.

class JuliaEmitter
{
  public:
    explicit JuliaEmitter(raw_ostream &os) : os(os) {}

    LogicalResult
    emitModule(ModuleOp module)
    {
	os << "# generated by ulm.lam (ulm-lam --emit=julia)\n";
	for (Operation &op : module.getBody()->getOperations()) {
	    if (auto v = dyn_cast<emitc::VerbatimOp>(op)) {
		StringRef text = v.getValue();
		if (text.starts_with("/*")) // the interface comment
		    os << "#=\n" << text << "\n=#\n";
		continue; // (C++ helpers: in UlmLAM.Kernels)
	    }
	    if (isa<emitc::IncludeOp>(op))
		continue;
	    if (auto f = dyn_cast<emitc::FuncOp>(op)) {
		if (failed(emitFunc(f)))
		    return failure();
		continue;
	    }
	    return op.emitError("ulm-lam --emit=julia: unexpected operation");
	}
	return success();
    }

  private:
    raw_ostream &os;
    llvm::DenseMap<Value, std::string> names;
    // emitc.subscript: the element p[i] as (pointer, index)
    llvm::DenseMap<Value, std::pair<std::string, std::string>> elements;
    unsigned counter = 0;
    unsigned depth = 1;

    std::string
    fresh()
    {
	return "v" + std::to_string(++counter);
    }

    std::string
    name(Value v)
    {
	auto it = names.find(v);
	if (it != names.end())
	    return it->second;
	return names[v] = fresh();
    }

    raw_ostream &
    line()
    {
	for (unsigned i = 0; i < depth; ++i)
	    os << "    ";
	return os;
    }

    /// The Julia type of an EmitC type.
    static std::string
    jtype(Type t)
    {
	if (auto l = dyn_cast<emitc::LValueType>(t))
	    return jtype(l.getValueType());
	if (auto p = dyn_cast<emitc::PointerType>(t))
	    return "Ptr{" + jtype(p.getPointee()) + "}";
	if (auto o = dyn_cast<emitc::OpaqueType>(t))
	    return o.getValue() == "void" ? "Cvoid" : "Int64";
	if (t.isF64())
	    return "Float64";
	if (t.isF32())
	    return "Float32";
	if (t.isInteger(1))
	    return "Bool";
	if (t.isInteger(8))
	    return "Int8";
	if (t.isInteger(32))
	    return "Int32";
	return "Int64"; // i64, index, size_t, ptrdiff_t
    }

    /// The C type of a parameter of an external function, as the
    /// runtime's declarations() parses it (the same as in C++).
    static std::string
    ctype(Type t)
    {
	if (auto p = dyn_cast<emitc::PointerType>(t))
	    return ctype(p.getPointee()) + " *";
	if (isa<emitc::SizeTType>(t))
	    return "size_t";
	if (t.isF64())
	    return "double";
	if (t.isF32())
	    return "float";
	if (t.isInteger(1))
	    return "bool";
	return "int64_t";
    }

    static bool
    isPointer(Type t)
    {
	return isa<emitc::PointerType>(t);
    }

    static bool
    isFloat(Type t)
    {
	return isa<FloatType>(t);
    }

    /// A Julia literal of an attribute (a constant, an argument of a call).
    std::string
    literal(Attribute a, Type type)
    {
	if (auto o = dyn_cast<emitc::OpaqueAttr>(a)) {
	    StringRef v = o.getValue();
	    if (v.empty())
		return zero(type);
	    return v.str();
	}
	if (auto f = dyn_cast<FloatAttr>(a)) {
	    double d = f.getValueAsDouble();
	    std::string s;
	    if (std::isnan(d))
		s = "NaN";
	    else if (std::isinf(d))
		s = d > 0 ? "Inf" : "-Inf";
	    else {
		llvm::raw_string_ostream ss(s);
		ss << llvm::format("%.17g", d);
		if (s.find_first_of(".eE") == std::string::npos)
		    s += ".0";
	    }
	    if (type && type.isF32())
		return "Float32(" + s + ")";
	    return s;
	}
	if (auto i = dyn_cast<IntegerAttr>(a)) {
	    if (i.getType().isInteger(1))
		return i.getValue().isZero() ? "false" : "true";
	    return std::to_string(i.getValue().getSExtValue());
	}
	if (auto b = dyn_cast<BoolAttr>(a))
	    return b.getValue() ? "true" : "false";
	return "";
    }

    static std::string
    zero(Type t)
    {
	std::string j = jtype(t);
	if (j.rfind("Ptr{", 0) == 0)
	    return j + "(C_NULL)";
	if (j == "Bool")
	    return "false";
	return "zero(" + j + ")";
    }

    LogicalResult
    emitFunc(emitc::FuncOp f)
    {
	FunctionType ft = f.getFunctionType();
	if (f.isExternal()) {
	    // an external function (bound with bind!): declared for the runtime
	    os << "# extern \"C\" void " << f.getName() << "(";
	    for (auto [k, t] : llvm::enumerate(ft.getInputs()))
		os << (k ? ", " : "") << ctype(t);
	    os << ");\n";
	    return success();
	}
	names.clear();
	elements.clear();
	os << "function lam_" << f.getName() << "(";
	Block &entry = f.getBody().front();
	for (auto [k, arg] : llvm::enumerate(entry.getArguments()))
	    os << (k ? ", " : "") << "a" << k + 1;
	os << ")\n";
	depth = 1;
	// the arguments of the ccall convention (Csize_t, Ptr{Cvoid}, ...)
	// as the types of the program
	for (auto [k, arg] : llvm::enumerate(entry.getArguments())) {
	    std::string t = jtype(arg.getType());
	    line() << name(arg) << " = " << t << "(a" << k + 1 << ")\n";
	}
	if (failed(emitBlock(entry)))
	    return failure();
	os << "end\n\n";
	return success();
    }

    LogicalResult
    emitRegion(Region &r)
    {
	if (r.empty())
	    return success();
	++depth;
	LogicalResult res = emitBlock(r.front());
	--depth;
	return res;
    }

    LogicalResult
    emitBlock(Block &b)
    {
	for (Operation &op : b)
	    if (failed(emitOp(op)))
		return failure();
	return success();
    }

    std::string
    operand(Operation &op, unsigned i)
    {
	return name(op.getOperand(i));
    }

    void
    assign(Operation &op, const std::string &expr)
    {
	line() << name(op.getResult(0)) << " = " << expr << "\n";
    }

    LogicalResult
    binary(Operation &op, StringRef jop)
    {
	assign(op, operand(op, 0) + " " + jop.str() + " " + operand(op, 1));
	return success();
    }

    /// The Julia function for a callee of emitc.call_opaque.
    static std::string
    callee(StringRef c)
    {
	c.consume_front("(void)");
	c = c.take_until([](char x) { return x == '<'; });
	if (c.consume_front("ulm_lam_cblas::") || c.consume_front("ulm_lam_"))
	    return "K." + c.str();
	if (c == "aligned_alloc" || c == "free")
	    return "K." + c.str();
	return c.str(); // an external function
    }

    LogicalResult
    emitOp(Operation &op)
    {
	if (auto c = dyn_cast<emitc::ConstantOp>(op)) {
	    assign(op, literal(c.getValue(), c.getType()));
	    return success();
	}
	if (auto v = dyn_cast<emitc::VariableOp>(op)) {
	    Type t = cast<emitc::LValueType>(v.getType()).getValueType();
	    std::string init = literal(v.getValue(), t);
	    assign(op, init.empty() ? zero(t) : init);
	    return success();
	}
	if (auto s = dyn_cast<emitc::SubscriptOp>(op)) {
	    if (s.getIndices().size() != 1 ||
	        !isPointer(s.getValue().getType()))
		return op.emitError("ulm-lam --emit=julia: subscript");
	    elements[s.getResult()] = {name(s.getValue()),
	                               name(s.getIndices()[0])};
	    return success();
	}
	if (auto l = dyn_cast<emitc::LoadOp>(op)) {
	    auto it = elements.find(l.getOperand());
	    if (it != elements.end())
		assign(op, "unsafe_load(" + it->second.first + ", " +
		               it->second.second + " + 1)");
	    else
		assign(op, name(l.getOperand()));
	    return success();
	}
	if (auto a = dyn_cast<emitc::AssignOp>(op)) {
	    auto it = elements.find(a.getVar());
	    if (it != elements.end())
		line() << "unsafe_store!(" << it->second.first << ", "
		       << name(a.getValue()) << ", " << it->second.second
		       << " + 1)\n";
	    else
		line() << name(a.getVar()) << " = " << name(a.getValue())
		       << "\n";
	    return success();
	}
	if (isa<emitc::AddOp>(op) && isPointer(op.getOperand(0).getType())) {
	    assign(op, "K.at(" + operand(op, 0) + ", " + operand(op, 1) + ")");
	    return success();
	}
	if (isa<emitc::SubOp>(op) && isPointer(op.getOperand(0).getType()))
	    return op.emitError("ulm-lam --emit=julia: pointer subtraction");
	if (isa<emitc::AddOp>(op))
	    return binary(op, "+");
	if (isa<emitc::SubOp>(op))
	    return binary(op, "-");
	if (isa<emitc::MulOp>(op))
	    return binary(op, "*");
	if (isa<emitc::DivOp>(op)) {
	    if (isFloat(op.getResult(0).getType()))
		return binary(op, "/");
	    assign(op, "div(" + operand(op, 0) + ", " + operand(op, 1) + ")");
	    return success();
	}
	if (isa<emitc::RemOp>(op)) {
	    assign(op, "rem(" + operand(op, 0) + ", " + operand(op, 1) + ")");
	    return success();
	}
	if (isa<emitc::BitwiseAndOp>(op))
	    return binary(op, "&");
	if (isa<emitc::BitwiseOrOp>(op))
	    return binary(op, "|");
	if (isa<emitc::BitwiseXorOp>(op))
	    return binary(op, "⊻");
	if (isa<emitc::LogicalAndOp>(op))
	    return binary(op, "&&");
	if (isa<emitc::LogicalOrOp>(op))
	    return binary(op, "||");
	if (isa<emitc::LogicalNotOp>(op)) {
	    assign(op, "!" + operand(op, 0));
	    return success();
	}
	if (isa<emitc::BitwiseNotOp>(op)) {
	    assign(op, "~" + operand(op, 0));
	    return success();
	}
	if (isa<emitc::UnaryMinusOp>(op)) {
	    assign(op, "-" + operand(op, 0));
	    return success();
	}
	if (isa<emitc::UnaryPlusOp>(op)) {
	    assign(op, operand(op, 0));
	    return success();
	}
	if (auto c = dyn_cast<emitc::CmpOp>(op)) {
	    const char *p = nullptr;
	    switch (c.getPredicate()) {
	    case emitc::CmpPredicate::eq:
		p = "==";
		break;
	    case emitc::CmpPredicate::ne:
		p = "!=";
		break;
	    case emitc::CmpPredicate::lt:
		p = "<";
		break;
	    case emitc::CmpPredicate::le:
		p = "<=";
		break;
	    case emitc::CmpPredicate::gt:
		p = ">";
		break;
	    case emitc::CmpPredicate::ge:
		p = ">=";
		break;
	    default:
		return op.emitError("ulm-lam --emit=julia: comparison");
	    }
	    return binary(op, p);
	}
	if (auto c = dyn_cast<emitc::ConditionalOp>(op)) {
	    assign(op, name(c.getCondition()) + " ? " + name(c.getTrueValue()) +
	                   " : " + name(c.getFalseValue()));
	    return success();
	}
	if (auto c = dyn_cast<emitc::CastOp>(op)) {
	    Type from = c.getSource().getType(), to = c.getType();
	    std::string v = name(c.getSource()), t = jtype(to);
	    if (isPointer(to))
		assign(op, t + "(" + v + ")");
	    else if (to.isInteger(1))
		assign(op, v + " != 0");
	    else if (isFloat(to))
		assign(op, t + "(" + v + ")");
	    else if (isFloat(from))
		assign(op, "trunc(" + t + ", " + v + ")");
	    else
		assign(op, t + "(" + v + ")");
	    return success();
	}
	if (auto c = dyn_cast<emitc::CallOpaqueOp>(op)) {
	    std::string args;
	    auto add = [&](const std::string &a) {
		args += (args.empty() ? "" : ", ") + a;
	    };
	    if (auto list = c.getArgs()) {
		for (Attribute a : *list) {
		    auto i = dyn_cast<IntegerAttr>(a);
		    if (i && i.getType().isIndex())
			add(operand(op, i.getInt()));
		    else
			add(literal(a, Type()));
		}
	    } else {
		for (unsigned i = 0; i < op.getNumOperands(); ++i)
		    add(operand(op, i));
	    }
	    std::string call = callee(c.getCallee()) + "(" + args + ")";
	    if (op.getNumResults() == 1)
		assign(op, call);
	    else
		line() << call << "\n";
	    return success();
	}
	if (auto c = dyn_cast<emitc::CallOp>(op)) {
	    std::string args;
	    for (unsigned i = 0; i < op.getNumOperands(); ++i)
		args += (i ? ", " : "") + operand(op, i);
	    std::string call = "lam_" + c.getCallee().str() + "(" + args + ")";
	    if (op.getNumResults() == 1)
		assign(op, call);
	    else if (op.getNumResults() == 0)
		line() << call << "\n";
	    else
		return op.emitError("ulm-lam --emit=julia: several results");
	    return success();
	}
	if (auto i = dyn_cast<emitc::IfOp>(op)) {
	    line() << "if " << name(i.getCondition()) << "\n";
	    if (failed(emitRegion(i.getThenRegion())))
		return failure();
	    if (!i.getElseRegion().empty()) {
		line() << "else\n";
		if (failed(emitRegion(i.getElseRegion())))
		    return failure();
	    }
	    line() << "end\n";
	    return success();
	}
	if (auto f = dyn_cast<emitc::ForOp>(op)) {
	    line() << "for " << name(f.getInductionVar()) << " in "
	           << name(f.getLowerBound()) << ":" << name(f.getStep())
	           << ":(" << name(f.getUpperBound()) << " - 1)\n";
	    if (failed(emitRegion(f.getRegion())))
		return failure();
	    line() << "end\n";
	    return success();
	}
	if (isa<emitc::YieldOp>(op))
	    return success();
	if (auto r = dyn_cast<emitc::ReturnOp>(op)) {
	    if (r.getOperand())
		line() << "return " << name(r.getOperand()) << "\n";
	    else
		line() << "return nothing\n";
	    return success();
	}
	if (auto v = dyn_cast<emitc::VerbatimOp>(op)) {
	    line() << "# " << v.getValue() << "\n";
	    return success();
	}
	return op.emitError("ulm-lam --emit=julia: unsupported operation");
    }
};

// ---------------------------------------------------------------------------
// The pipeline (as scripts/ulm-lam-compile.sh)
// ---------------------------------------------------------------------------

LogicalResult
runPipeline(ModuleOp module, const std::string &pipeline)
{
    // (the pipeline is anchored on builtin.module, as the module itself)
    auto parsed = parsePassPipeline(pipeline, llvm::errs());
    if (failed(parsed))
	return failure();
    PassManager pm(module->getName());
    std::string err;
    llvm::raw_string_ostream es(err);
    if (failed(parsePassPipeline(
            StringRef(pipeline)
                .drop_front(StringRef("builtin.module(").size())
                .drop_back(),
            pm, es))) {
	llvm::errs() << "ulm-lam: bad pipeline: " << err << "\n";
	return failure();
    }
    return pm.run(module);
}

LogicalResult
writeFile(const std::string &path, llvm::function_ref<void(raw_ostream &)> f)
{
    std::string err;
    auto out = openOutputFile(path, &err);
    if (!out) {
	llvm::errs() << "ulm-lam: " << err << "\n";
	return failure();
    }
    f(out->os());
    out->keep();
    return success();
}

} // namespace

int
main(int argc, char **argv)
{
    llvm::InitLLVM init(argc, argv);
    // only what the three pipelines use (not registerAll*: that would link
    // every dialect and every LLVM target into the executable)
    registerTransformsPasses();
    bufferization::registerBufferizationPasses();
    bufferization::registerBufferizationPipelines();
    registerReconcileUnrealizedCastsPass();
    emitc::registerEmitCPasses();
    ::ulm_lam::registerResultsToOutParamsPass();
    ::ulm_lam::registerConvertUlmLamkToUlmBLASPass();
    ::ulm_lam::registerUlmLamPasses();
    ::ulm_lam::registerWorkspacePass();
    ::ulm_lam::registerInfoPass();
    cl::ParseCommandLineOptions(argc, argv,
                                "ulm.lam: linear algebra onto BLAS and "
                                "LAPACK (ulm_lam IR to C++ or Julia)\n");
    if (backend != "ulmblas" && backend != "cblas") {
	llvm::errs() << "ulm-lam: --backend must be ulmblas or cblas\n";
	return 1;
    }
    if (emit != "cpp" && emit != "julia") {
	llvm::errs() << "ulm-lam: --emit must be cpp or julia\n";
	return 1;
    }

    DialectRegistry registry;
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    cf::ControlFlowDialect, emitc::EmitCDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect, ub::UBDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
    arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    cf::registerBufferizableOpInterfaceExternalModels(registry);
    cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    memref::registerAllocationOpInterfaceExternalModels(registry);
    memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerSubsetOpInterfaceExternalModels(registry);
    registry
        .insert<mlir::ulm_lam::UlmLamDialect, mlir::ulm_lamk::UlmLamkDialect>();
    mlir::ulm_lam::registerBufferizationModels(registry);
    MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    llvm::SourceMgr sm;
    SourceMgrDiagnosticHandler diag(sm, &ctx);
    std::string err;
    auto file = openInputFile(inputFile, &err);
    if (!file) {
	llvm::errs() << "ulm-lam: " << err << "\n";
	return 1;
    }
    sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &ctx);
    if (!module)
	return 1;

    bool lapack = backend == "cblas";
    std::string select =
        "builtin.module(func.func(ulm-lam-normalize, "
        "ulm-lam-infer-properties" +
        std::string(noLicm ? "" : ", loop-invariant-code-motion") +
        ", ulm-lam-select-kernels{lapack=" + (lapack ? "true" : "false") + " " +
        selectOptions + "}), ulm-lam-strip-properties, canonicalize)";
    std::string bufferize =
        "builtin.module(ulm-lam-results-to-out-params, "
        "eliminate-empty-tensors, "
        "one-shot-bufferize{bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map}, "
        "buffer-results-to-out-params{hoist-static-allocs "
        "modify-public-functions add-result-attr}, "
        "func.func(buffer-loop-hoisting), buffer-deallocation-pipeline, "
        "canonicalize, ulm-lam-workspace, canonicalize, cse, ulm-lam-info, "
        "canonicalize)";
    std::string lower =
        "builtin.module(convert-ulm-lamk-to-ulmblas{backend=" + backend +
        "}, reconcile-unrealized-casts" +
        (emit == "cpp" ? ", form-expressions" : "") + ")";

    auto stage = [&](const char *what) -> LogicalResult {
	if (stagesPrefix.empty())
	    return success();
	return writeFile(stagesPrefix + "." + what + ".mlir",
	                 [&](raw_ostream &os) { module->print(os); });
    };
    if (failed(runPipeline(*module, select)) || failed(stage("selected")))
	return 1;
    if (!selectedFile.empty() &&
        failed(writeFile(selectedFile,
                         [&](raw_ostream &os) { module->print(os); })))
	return 1;
    if (failed(runPipeline(*module, bufferize)) || failed(stage("bufferized")))
	return 1;
    if (failed(runPipeline(*module, lower)) || failed(stage("emitc")))
	return 1;

    LogicalResult res = success();
    if (failed(writeFile(outputFile, [&](raw_ostream &os) {
	    res = emit == "cpp" ? emitc::translateToCpp(*module, os)
	                        : JuliaEmitter(os).emitModule(*module);
        })))
	return 1;
    return failed(res) ? 1 : 0;
}
