//===- ulm-lam.cpp - the ulm.lam compiler: ulm_lam IR to C++, Julia, Mojo -===//
//
//   ulm-lam [--backend=ulmblas|cblas] [--emit=cpp|julia|mojo] [--mojo-c-abi]
//           [--select-options=...] [--no-licm] [--selected=FILE]
//           input.mlir -o output
//
// The pipeline of scripts/ulm-lam-compile.sh in one process, without
// mlir-opt, a plugin or mlir-translate: selection of the kernels,
// bufferization, lowering to EmitC, and then C++ (EmitC's translation, as
// mlir-translate --mlir-to-cpp), Julia (JuliaEmitter below: the same
// program calling the kernels of UlmLAM.Kernels, so that Julia needs no C++
// compiler) or Mojo (MojoEmitter: the kernels of mojo/ulm_lam/kernels.mojo
// on Apple Accelerate).  The frontends run this tool; linked statically
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
#include "llvm/ADT/StringSet.h"
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
static cl::opt<std::string> emit("emit", cl::desc("cpp, julia or mojo"),
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
// Mojo: the EmitC program as a Mojo module
// ---------------------------------------------------------------------------
//
// As for Julia, every EmitC operation becomes one statement: a typed local
// variable (var vN: T = ...), C's lvalues are local variables
// (emitc.variable, declared without a value when EmitC has none) or
// pointer elements (emitc.subscript: p[unsafe_offset=i]).  Integers of 64
// bits (index, size_t, ptrdiff_t, i64) are Int; pointers are typed
// (P[DType.float64], P = ulm_lam.kernels.P), pointer arithmetic counts
// elements (unsafe_offset).  The kernels are the functions of
// ulm_lam.kernels (mojo/ulm_lam/kernels.mojo), imported as K.
//
// External functions (the caller's) are the methods of a trait Externs:
// every function of the program takes the caller's implementation as its
// first argument (mut ext: E, E: Externs).  With --mojo-c-abi, every
// function is also exported under its name with the C calling convention
// (abi("C")), and the external functions are called as C functions (the
// struct CExterns): the shared library a C caller or Python's ctypes
// loads, as the library of the generated C++.

static cl::opt<bool>
    mojoCAbi("mojo-c-abi",
             cl::desc("--emit=mojo: export the functions with the C calling "
                      "convention, external functions as C functions"),
             cl::init(false));

class MojoEmitter
{
  public:
    explicit MojoEmitter(raw_ostream &os) : os(os) {}

    LogicalResult
    emitModule(ModuleOp module)
    {
	os << "# generated by ulm.lam (ulm-lam --emit=mojo)\n";
	if (mojoCAbi)
	    os << "from std.ffi import external_call\n";
	os << "from ulm_lam import kernels as K\n"
	   << "from ulm_lam.kernels import P\n";
	SmallVector<emitc::FuncOp> externals, functions;
	for (Operation &op : module.getBody()->getOperations()) {
	    if (isa<emitc::VerbatimOp, emitc::IncludeOp>(op))
		continue;
	    auto f = dyn_cast<emitc::FuncOp>(op);
	    if (!f)
		return op.emitError(
		    "ulm-lam --emit=mojo: unexpected operation");
	    (f.isExternal() ? externals : functions).push_back(f);
	    if (f.isExternal())
		externalNames.insert(f.getName());
	}
	hasExterns = !externals.empty();
	if (hasExterns) {
	    os << "\n\ntrait Externs:\n"
	       << "    \"\"\"The external functions of the program, "
	          "implemented "
	          "by the caller.\"\"\"\n";
	    for (emitc::FuncOp f : externals) {
		os << "\n    def ";
		externSignature(f);
		os << ": ...\n";
	    }
	}
	if (hasExterns && mojoCAbi) {
	    os << "\n\nstruct CExterns(Externs):\n"
	       << "    \"\"\"The external functions as C functions (of the "
	          "caller of the C interface).\"\"\"\n\n"
	       << "    def __init__(out self):\n        pass\n";
	    for (emitc::FuncOp f : externals) {
		os << "\n    def ";
		externSignature(f);
		os << ":\n        external_call[\"" << f.getName()
		   << "\", NoneType](";
		for (unsigned k = 0; k < f.getNumArguments(); ++k)
		    os << (k ? ", " : "") << "a" << k + 1;
		os << ")\n";
	    }
	}
	// the interface comments, by function
	llvm::StringMap<std::string> docs;
	for (Operation &op : module.getBody()->getOperations())
	    if (auto v = dyn_cast<emitc::VerbatimOp>(op)) {
		StringRef text = v.getValue();
		if (!text.starts_with("/* "))
		    continue;
		StringRef fn = text.drop_front(3).take_until(
		    [](char c) { return c == ':'; });
		docs[fn] = text.str();
	    }
	for (emitc::FuncOp f : functions) {
	    os << "\n\n";
	    auto it = docs.find(f.getName());
	    if (it != docs.end())
		comment(it->second);
	    if (failed(emitFunc(f)))
		return failure();
	}
	if (mojoCAbi)
	    for (emitc::FuncOp f : functions)
		exportFunc(f);
	return success();
    }

  private:
    raw_ostream &os;
    llvm::DenseMap<Value, std::string> names;
    // emitc.subscript: the element p[i] as (pointer, index, pointer type)
    struct Element
    {
	std::string ptr, index;
	Type pointee;
    };
    llvm::DenseMap<Value, Element> elements;
    llvm::StringSet<> externalNames;
    bool hasExterns = false;
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

    /// The C interface comment (/* ... */) as Mojo comment lines.
    void
    comment(StringRef text)
    {
	SmallVector<StringRef> lines;
	text.split(lines, '\n');
	for (StringRef l : lines) {
	    l = l.rtrim();
	    if (l.consume_front("/*"))
		l = l.ltrim();
	    else if (l.consume_front(" */") || l == "*/")
		continue;
	    else if (l.consume_front(" *"))
		l.consume_front(" ");
	    os << (l.empty() ? "#" : "# " + l.str()) << "\n";
	}
    }

    /// The element type of a pointer, as a Mojo DType.
    static std::string
    dtype(Type t)
    {
	if (auto l = dyn_cast<emitc::LValueType>(t))
	    return dtype(l.getValueType());
	if (t.isF64())
	    return "DType.float64";
	if (t.isF32())
	    return "DType.float32";
	if (t.isInteger(1))
	    return "DType.bool";
	if (t.isInteger(8) || isa<emitc::OpaqueType>(t)) // (void *: bytes)
	    return "DType.int8";
	if (t.isInteger(32))
	    return "DType.int32";
	return "DType.int64"; // i64, index, size_t, ptrdiff_t
    }

    /// The Mojo type of an EmitC type.
    static std::string
    mtype(Type t)
    {
	if (auto l = dyn_cast<emitc::LValueType>(t))
	    return mtype(l.getValueType());
	if (auto p = dyn_cast<emitc::PointerType>(t))
	    return "P[" + dtype(p.getPointee()) + "]";
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
	return "Int"; // i64, index, size_t, ptrdiff_t
    }

    static bool
    isPointer(Type t)
    {
	if (auto l = dyn_cast<emitc::LValueType>(t))
	    t = l.getValueType();
	return isa<emitc::PointerType>(t);
    }

    static bool
    isFloat(Type t)
    {
	return isa<FloatType>(t);
    }

    /// A Mojo literal of an attribute; typed (Float64(1.0)) where the
    /// type is not given by a declaration (an argument of a call).
    std::string
    literal(Attribute a, bool typed)
    {
	if (auto o = dyn_cast<emitc::OpaqueAttr>(a)) {
	    StringRef v = o.getValue();
	    if (v == "true")
		return "True";
	    if (v == "false")
		return "False";
	    return v.str();
	}
	if (auto f = dyn_cast<FloatAttr>(a)) {
	    double d = f.getValueAsDouble();
	    std::string t = mtype(f.getType()), s;
	    if (std::isnan(d))
		return t + ".nan()";
	    if (std::isinf(d))
		return d > 0 ? t + ".inf()" : "-" + t + ".inf()";
	    llvm::raw_string_ostream ss(s);
	    ss << llvm::format("%.17g", d);
	    if (s.find_first_of(".eEn") == std::string::npos)
		s += ".0";
	    return typed ? t + "(" + s + ")" : s;
	}
	if (auto i = dyn_cast<IntegerAttr>(a)) {
	    if (i.getType().isInteger(1))
		return i.getValue().isZero() ? "False" : "True";
	    return std::to_string(i.getValue().getSExtValue());
	}
	if (auto b = dyn_cast<BoolAttr>(a))
	    return b.getValue() ? "True" : "False";
	return "";
    }

    /// name: T for the parameters of a signature (a1, a2, ...).
    void
    externSignature(emitc::FuncOp f)
    {
	os << f.getName() << "(mut self";
	for (auto [k, t] : llvm::enumerate(f.getArgumentTypes()))
	    os << ", a" << k + 1 << ": " << mtype(t);
	os << ")";
	FunctionType ft = f.getFunctionType();
	if (ft.getNumResults() == 1)
	    os << " -> " << mtype(ft.getResult(0));
    }

    /// The parameter list of a function of the program.
    std::string
    parameters(emitc::FuncOp f, bool withExterns)
    {
	std::string s = withExterns ? "mut ext: E" : "";
	Block &entry = f.getBody().front();
	for (BlockArgument arg : entry.getArguments())
	    s += (s.empty() ? "" : ", ") + name(arg) + ": " +
	         mtype(arg.getType());
	return s;
    }

    std::string
    resultType(emitc::FuncOp f)
    {
	FunctionType ft = f.getFunctionType();
	return ft.getNumResults() == 1 ? " -> " + mtype(ft.getResult(0)) : "";
    }

    LogicalResult
    emitFunc(emitc::FuncOp f)
    {
	names.clear();
	elements.clear();
	counter = 0;
	os << "def " << f.getName() << (hasExterns ? "[E: Externs]" : "") << "("
	   << parameters(f, hasExterns) << ")" << resultType(f) << ":\n";
	depth = 1;
	return emitBlock(f.getBody().front());
    }

    /// The C entry point of a function (--mojo-c-abi).
    void
    exportFunc(emitc::FuncOp f)
    {
	names.clear();
	counter = 0;
	std::string params = parameters(f, false);
	std::string args = hasExterns ? "ext" : "";
	for (BlockArgument arg : f.getBody().front().getArguments())
	    args += (args.empty() ? "" : ", ") + name(arg);
	os << "\n\n@export(\"" << f.getName() << "\")\n"
	   << "def " << f.getName() << "_c_abi(" << params << ") abi(\"C\")"
	   << resultType(f) << ":\n";
	if (hasExterns)
	    os << "    var ext = CExterns()\n";
	os << "    " << (f.getFunctionType().getNumResults() ? "return " : "")
	   << f.getName() << "(" << args << ")\n";
    }

    LogicalResult
    emitRegion(Region &r)
    {
	++depth;
	bool empty = true;
	if (!r.empty())
	    for (Operation &op : r.front())
		if (!isa<emitc::YieldOp>(op))
		    empty = false;
	LogicalResult res = success();
	if (empty)
	    line() << "pass\n";
	else
	    res = emitBlock(r.front());
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
	Value r = op.getResult(0);
	line() << "var " << name(r) << ": " << mtype(r.getType()) << " = "
	       << expr << "\n";
    }

    LogicalResult
    binary(Operation &op, StringRef mop)
    {
	assign(op, operand(op, 0) + " " + mop.str() + " " + operand(op, 1));
	return success();
    }

    /// The Mojo function for a callee of emitc.call_opaque.
    std::string
    callee(StringRef c)
    {
	c.consume_front("(void)");
	c = c.take_until([](char x) { return x == '<'; });
	if (c.consume_front("ulm_lam_cblas::") || c.consume_front("ulm_lam_"))
	    return "K." + c.str();
	if (c == "aligned_alloc" || c == "free")
	    return "K." + c.str();
	return "ext." + c.str(); // an external function
    }

    std::string
    castExpr(emitc::CastOp c)
    {
	Type from = c.getSource().getType(), to = c.getType();
	std::string v = name(c.getSource()), t = mtype(to);
	if (mtype(from) == t)
	    return v;
	if (isPointer(to)) {
	    auto p = cast<emitc::PointerType>(to);
	    if (isPointer(from))
		return v + ".unsafe_bitcast[Scalar[" + dtype(p.getPointee()) +
		       "]]()";
	    return t + "(unsafe_from_address=Int(" + v + "))";
	}
	if (to.isInteger(1))
	    return v + " != 0";
	return t + "(" + v + ")";
    }

    LogicalResult
    emitOp(Operation &op)
    {
	if (auto c = dyn_cast<emitc::ConstantOp>(op)) {
	    assign(op, literal(c.getValue(), false));
	    return success();
	}
	if (auto v = dyn_cast<emitc::VariableOp>(op)) {
	    Type t = cast<emitc::LValueType>(v.getType()).getValueType();
	    std::string init = literal(v.getValue(), false);
	    line() << "var " << name(v.getResult()) << ": " << mtype(t)
	           << (init.empty() ? "" : " = " + init) << "\n";
	    return success();
	}
	if (auto s = dyn_cast<emitc::SubscriptOp>(op)) {
	    if (s.getIndices().size() != 1 ||
	        !isPointer(s.getValue().getType()))
		return op.emitError("ulm-lam --emit=mojo: subscript");
	    auto p = cast<emitc::PointerType>(s.getValue().getType());
	    elements[s.getResult()] = {name(s.getValue()),
	                               name(s.getIndices()[0]), p.getPointee()};
	    return success();
	}
	if (auto l = dyn_cast<emitc::LoadOp>(op)) {
	    auto it = elements.find(l.getOperand());
	    if (it != elements.end())
		assign(op, mtype(l.getType()) + "(" + it->second.ptr +
		               "[unsafe_offset=" + it->second.index + "])");
	    else
		assign(op, name(l.getOperand()));
	    return success();
	}
	if (auto a = dyn_cast<emitc::AssignOp>(op)) {
	    auto it = elements.find(a.getVar());
	    if (it != elements.end())
		line() << it->second.ptr
		       << "[unsafe_offset=" << it->second.index << "] = Scalar["
		       << dtype(it->second.pointee) << "]("
		       << name(a.getValue()) << ")\n";
	    else
		line() << name(a.getVar()) << " = " << name(a.getValue())
		       << "\n";
	    return success();
	}
	if (isa<emitc::AddOp>(op) && isPointer(op.getOperand(0).getType())) {
	    assign(op,
	           operand(op, 0) + ".unsafe_offset(" + operand(op, 1) + ")");
	    return success();
	}
	if (isa<emitc::SubOp>(op) && isPointer(op.getOperand(0).getType()))
	    return op.emitError("ulm-lam --emit=mojo: pointer subtraction");
	if (isa<emitc::AddOp>(op))
	    return binary(op, "+");
	if (isa<emitc::SubOp>(op))
	    return binary(op, "-");
	if (isa<emitc::MulOp>(op))
	    return binary(op, "*");
	if (isa<emitc::DivOp>(op)) {
	    if (isFloat(op.getResult(0).getType()))
		return binary(op, "/");
	    // (C's division truncates, Mojo's // floors)
	    assign(op,
	           "K.cdiv(" + operand(op, 0) + ", " + operand(op, 1) + ")");
	    return success();
	}
	if (isa<emitc::RemOp>(op)) {
	    assign(op,
	           "K.crem(" + operand(op, 0) + ", " + operand(op, 1) + ")");
	    return success();
	}
	if (isa<emitc::BitwiseAndOp>(op))
	    return binary(op, "&");
	if (isa<emitc::BitwiseOrOp>(op))
	    return binary(op, "|");
	if (isa<emitc::BitwiseXorOp>(op))
	    return binary(op, "^");
	if (isa<emitc::LogicalAndOp>(op))
	    return binary(op, "and");
	if (isa<emitc::LogicalOrOp>(op))
	    return binary(op, "or");
	if (isa<emitc::LogicalNotOp>(op)) {
	    assign(op, "not " + operand(op, 0));
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
		return op.emitError("ulm-lam --emit=mojo: comparison");
	    }
	    return binary(op, p);
	}
	if (auto c = dyn_cast<emitc::ConditionalOp>(op)) {
	    assign(op, name(c.getTrueValue()) + " if " +
	                   name(c.getCondition()) + " else " +
	                   name(c.getFalseValue()));
	    return success();
	}
	if (auto c = dyn_cast<emitc::CastOp>(op)) {
	    // (a cast that does not change the Mojo type: the same variable)
	    if (mtype(c.getSource().getType()) == mtype(c.getType())) {
		names[c.getResult()] = name(c.getSource());
		return success();
	    }
	    assign(op, castExpr(c));
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
			add(literal(a, true));
		}
	    } else {
		for (unsigned i = 0; i < op.getNumOperands(); ++i)
		    add(operand(op, i));
	    }
	    std::string call = callee(c.getCallee()) + "(" + args + ")";
	    if (op.getNumResults() == 1)
		assign(op, call);
	    else
		line() << "_ = " << call << "\n";
	    return success();
	}
	if (auto c = dyn_cast<emitc::CallOp>(op)) {
	    StringRef fn = c.getCallee();
	    bool external = externalNames.contains(fn);
	    std::string args = external || !hasExterns ? "" : "ext";
	    for (unsigned i = 0; i < op.getNumOperands(); ++i)
		args += (args.empty() ? "" : ", ") + operand(op, i);
	    std::string call =
	        (external ? "ext." : "") + fn.str() + "(" + args + ")";
	    if (op.getNumResults() == 1)
		assign(op, call);
	    else if (op.getNumResults() == 0)
		line() << call << "\n";
	    else
		return op.emitError("ulm-lam --emit=mojo: several results");
	    return success();
	}
	if (auto i = dyn_cast<emitc::IfOp>(op)) {
	    line() << "if " << name(i.getCondition()) << ":\n";
	    if (failed(emitRegion(i.getThenRegion())))
		return failure();
	    if (!i.getElseRegion().empty()) {
		line() << "else:\n";
		if (failed(emitRegion(i.getElseRegion())))
		    return failure();
	    }
	    return success();
	}
	if (auto f = dyn_cast<emitc::ForOp>(op)) {
	    line() << "for " << name(f.getInductionVar()) << " in range("
	           << name(f.getLowerBound()) << ", " << name(f.getUpperBound())
	           << ", " << name(f.getStep()) << "):\n";
	    return emitRegion(f.getRegion());
	}
	if (isa<emitc::YieldOp>(op))
	    return success();
	if (auto r = dyn_cast<emitc::ReturnOp>(op)) {
	    if (r.getOperand())
		line() << "return " << name(r.getOperand()) << "\n";
	    else
		line() << "return\n";
	    return success();
	}
	if (auto v = dyn_cast<emitc::VerbatimOp>(op)) {
	    line() << "# " << v.getValue() << "\n";
	    return success();
	}
	return op.emitError("ulm-lam --emit=mojo: unsupported operation");
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
                                "LAPACK (ulm_lam IR to C++, Julia or Mojo)\n");
    if (backend != "ulmblas" && backend != "cblas") {
	llvm::errs() << "ulm-lam: --backend must be ulmblas or cblas\n";
	return 1;
    }
    if (emit != "cpp" && emit != "julia" && emit != "mojo") {
	llvm::errs() << "ulm-lam: --emit must be cpp, julia or mojo\n";
	return 1;
    }
    if (emit == "mojo" && backend != "cblas") {
	llvm::errs() << "ulm-lam: --emit=mojo needs --backend=cblas (the "
	                "kernels of ulm_lam.kernels)\n";
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
        (emit == "cpp" ? ", form-expressions, ulm-lam-call-statements" : "") +
        ")";

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
	    if (emit == "cpp")
	        res = emitc::translateToCpp(*module, os);
	    else if (emit == "julia")
	        res = JuliaEmitter(os).emitModule(*module);
	    else
	        res = MojoEmitter(os).emitModule(*module);
        })))
	return 1;
    return failed(res) ? 1 : 0;
}
