//===- ulm-lam-opt.cpp - mlir-opt with the ulm.lam dialects and passes ----===//
//
//   ulm-lam-opt input.mlir --pass-pipeline='builtin.module(func.func(
//       ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))'
//
// The same as mlir-opt with the plugin loaded (build/lib/UlmLamPasses), as
// one tool: the lit tests in test/ run it.
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

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

int
main(int argc, char **argv)
{
    mlir::registerAllPasses();
    ulm_lam::registerResultsToOutParamsPass();
    ulm_lam::registerConvertUlmLamkToUlmBLASPass();
    ulm_lam::registerUlmLamPasses();
    ulm_lam::registerWorkspacePass();
    ulm_lam::registerInfoPass();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    registry
        .insert<mlir::ulm_lam::UlmLamDialect, mlir::ulm_lamk::UlmLamkDialect>();
    mlir::ulm_lam::registerBufferizationModels(registry);
    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "ulm.lam: linear algebra onto BLAS and LAPACK\n",
        registry));
}
