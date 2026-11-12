//===- Plugin.cpp - mlir-opt plugin entry points --------------------------===//
//
//   mlir-opt --load-dialect-plugin=build/lib/UlmLamPasses.dylib \
//            --load-pass-plugin=build/lib/UlmLamPasses.dylib \
//            --pass-pipeline='builtin.module(ulm-lam-select-kernels, ...)'
//
//===----------------------------------------------------------------------===//

#include "UlmLamDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

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

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo()
{
    return {MLIR_PLUGIN_API_VERSION, "UlmLamDialects", LLVM_VERSION_STRING,
            [](mlir::DialectRegistry *registry) {
	        registry->insert<mlir::ulm_lam::UlmLamDialect,
		                 mlir::ulm_lamk::UlmLamkDialect>();
	        mlir::ulm_lam::registerBufferizationModels(*registry);
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo()
{
    return {MLIR_PLUGIN_API_VERSION, "UlmLamPasses", LLVM_VERSION_STRING, [] {
	        ulm_lam::registerResultsToOutParamsPass();
	        ulm_lam::registerConvertUlmLamkToUlmBLASPass();
	        ulm_lam::registerUlmLamPasses();
	        ulm_lam::registerWorkspacePass();
	        ulm_lam::registerInfoPass();
            }};
}
