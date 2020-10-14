# For a better template to copy, see examples/standalone
add_subdirectory(inc)
include_directories(inc)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  X86
  AArch64

  # all
  # MC
  # MCParser
  # Object
  # Support
  # Symbolize
  # X86
  # Target
  # ScalarOpts
  # Support
  # native
  # Analysis
  # Core
  # ExecutionEngine
  # InstCombine
  # Object
  # OrcJIT
  # RuntimeDyld
  # ScalarOpts
  # Support
  # TransformUtils
  )

add_toy_chapter(es2dsl
  es2dslmain.cpp
  src/dialect/es2tlv_dialect.cpp
  src/dialect/es2tlv_ops.cpp
  src/dialect/es2tlv_lowertocpp.cpp
  src/dialect/es2tlv_canonical.cpp
  src/dialect/es2tlv_lowertoaffine.cpp
  src/dialect/es2tlv_shapeinfer.cpp
  src/subsys/es2mlirsubsys.cpp
  src/subsys/es2mlirsubsys_mlirgen.cpp
  )


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(es2dsl
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRPass
    MLIRTranslation
    MLIRSupport
    MLIREmitC
    MLIRTargetCpp)
