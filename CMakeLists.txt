# The ulm.lam compiler as one executable (tools/ulm-lam.cpp): ulm_lam IR
# (intermediate representation) to C++ or Julia.  This is the build of the
# binary package ulm_lam_jll; for development, scripts/build-driver.sh and
# scripts/build-plugin.sh build the same from Homebrew's LLVM (Low Level
# Virtual Machine) without CMake.
#
#   cmake -S . -B build/cmake -DMLIR_DIR=<prefix>/lib/cmake/mlir
#   cmake --build build/cmake && cmake --install build/cmake --prefix <dir>
#
# Installs bin/ulm-lam and include/ulm_lam/cblas_backend.h (the header the
# generated C++ includes for the CBLAS backend).

cmake_minimum_required(VERSION 3.20)
project(ulm-lam LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "MLIR (Multi-Level Intermediate Representation): ${MLIR_DIR}")
# ULM_LAM_TABLEGEN: an mlir-tblgen other than MLIR's own (the build
# machine's when cross-compiling; MLIRConfig.cmake sets MLIR_TABLEGEN_EXE)
set(ULM_LAM_TABLEGEN "" CACHE FILEPATH "mlir-tblgen to run")
if(ULM_LAM_TABLEGEN)
  set(MLIR_TABLEGEN_EXE ${ULM_LAM_TABLEGEN})
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# ULM_LAM_STATIC_LLVM: no libLLVM at run time (the binary package must not
# depend on the libLLVM of a particular Julia).  The exported MLIR targets
# link "LLVM", the shared library, when LLVM was built with
# LLVM_LINK_LLVM_DYLIB (Homebrew, Yggdrasil); redirect them to LLVM's
# static component libraries.
option(ULM_LAM_STATIC_LLVM "link LLVM's static libraries" OFF)
set(ulm_lam_exe_flags)
if(ULM_LAM_STATIC_LLVM)
  set(ulm_lam_exe_flags DISABLE_LLVM_LINK_LLVM_DYLIB)
  llvm_map_components_to_libnames(ulm_lam_llvm_libs Support Demangle)
  add_library(ulm_lam_llvm_static INTERFACE)
  target_link_libraries(ulm_lam_llvm_static INTERFACE ${ulm_lam_llvm_libs})
  # (only their libraries: zlib's include directory, the SDK's on macOS,
  # would come before libc++'s headers)
  foreach(t ZLIB::ZLIB zstd::libzstd_shared zstd::libzstd_static)
    if(TARGET ${t})
      set_target_properties(${t} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    endif()
  endforeach()
  foreach(t IN LISTS MLIR_EXPORTED_TARGETS)
    if(TARGET ${t})
      get_target_property(deps ${t} INTERFACE_LINK_LIBRARIES)
      if(deps)
        list(TRANSFORM deps REPLACE "^LLVM$" "ulm_lam_llvm_static")
        set_target_properties(${t} PROPERTIES INTERFACE_LINK_LIBRARIES "${deps}")
      endif()
    endif()
  endforeach()
endif()

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib
                    ${PROJECT_BINARY_DIR})

# C++ declarations from include/ulm_lam/*.td (as in scripts/build-plugin.sh)
function(ulm_lam_tablegen td)
  set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/ulm_lam/${td})
  set(args ${ARGN})
  while(args)
    list(POP_FRONT args out backend extra)
    if(extra STREQUAL "-")
      mlir_tablegen(${out} ${backend})
    else()
      mlir_tablegen(${out} ${backend} ${extra})
    endif()
    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} PARENT_SCOPE)
  endwhile()
endfunction()

ulm_lam_tablegen(UlmLamBase.td
  UlmLamDialect.h.inc -gen-dialect-decls -dialect=ulm_lam
  UlmLamDialect.cpp.inc -gen-dialect-defs -dialect=ulm_lam
  UlmLamkDialect.h.inc -gen-dialect-decls -dialect=ulm_lamk
  UlmLamkDialect.cpp.inc -gen-dialect-defs -dialect=ulm_lamk)
ulm_lam_tablegen(UlmLamAttrs.td
  UlmLamEnums.h.inc -gen-enum-decls -
  UlmLamEnums.cpp.inc -gen-enum-defs -
  UlmLamAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ulm_lam
  UlmLamAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ulm_lam
  UlmLamkAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ulm_lamk
  UlmLamkAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ulm_lamk)
ulm_lam_tablegen(UlmLamInterfaces.td
  UlmLamInterfaces.h.inc -gen-op-interface-decls -
  UlmLamInterfaces.cpp.inc -gen-op-interface-defs -)
ulm_lam_tablegen(UlmLamOps.td
  UlmLamOps.h.inc -gen-op-decls -
  UlmLamOps.cpp.inc -gen-op-defs -)
ulm_lam_tablegen(UlmLamKernelOps.td
  UlmLamkOps.h.inc -gen-op-decls -
  UlmLamkOps.cpp.inc -gen-op-defs -)
add_public_tablegen_target(UlmLamIncGen)

# lib/ without Plugin.cpp (the entry points of the mlir-opt plugin)
add_llvm_executable(ulm-lam ${ulm_lam_exe_flags}
  tools/ulm-lam.cpp
  lib/ConvertUlmLamkToUlmBLAS.cpp
  lib/UlmLamDialect.cpp
  lib/UlmLamInfo.cpp
  lib/UlmLamkDialect.cpp
  lib/UlmLamPasses.cpp
  lib/UlmLamResultsToOutParams.cpp
  lib/UlmLamWorkspace.cpp
  DEPENDS UlmLamIncGen)
llvm_update_compile_flags(ulm-lam)
target_link_libraries(ulm-lam PRIVATE
  MLIRArithDialect MLIRArithToEmitC MLIRArithTransforms
  MLIRBufferizationDialect MLIRBufferizationPipelines
  MLIRBufferizationTransforms MLIRControlFlowDialect
  MLIRControlFlowTransforms MLIREmitCDialect MLIREmitCTransforms
  MLIRFuncDialect MLIRFuncToEmitC MLIRIR MLIRMemRefDialect
  MLIRMemRefTransforms MLIRParser MLIRPass MLIRReconcileUnrealizedCasts
  MLIRSCFDialect MLIRSCFToEmitC MLIRSCFTransforms MLIRSupport
  MLIRTargetCpp MLIRTensorDialect MLIRTensorTransforms MLIRTransforms
  MLIRTransformUtils MLIRUBDialect)

if(APPLE)
  # (LLVMSupport names z3 and zstd; the compiler uses neither)
  target_link_options(ulm-lam PRIVATE -Wl,-dead_strip -Wl,-dead_strip_dylibs)
endif()

install(TARGETS ulm-lam RUNTIME DESTINATION bin)
install(FILES include/ulm_lam/cblas_backend.h DESTINATION include/ulm_lam)
