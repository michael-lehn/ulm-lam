//===- UlmLamkDialect.cpp - ulm_lamk dialect and attributes ---------------===//
//
// Separate translation unit: the generated attribute parser/printer helpers
// of ulm_lam and ulm_lamk have the same (static) names.
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "UlmLamkDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "UlmLamkAttrs.cpp.inc"

void
mlir::ulm_lamk::UlmLamkDialect::initialize()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "UlmLamkAttrs.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "UlmLamkOps.cpp.inc"
        >();
}
