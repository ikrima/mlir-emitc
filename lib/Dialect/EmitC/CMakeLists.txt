add_mlir_dialect_library(MLIREmitC
  IR/EmitCDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/emitc/Dialect/EmitC

  DEPENDS
  MLIREmitCIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
