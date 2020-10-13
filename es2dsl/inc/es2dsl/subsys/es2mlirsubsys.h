//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//
#pragma  once

#include <memory>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
} // namespace mlir

namespace es2 {
  struct Module_ast;

  std::unique_ptr<Module_ast> astGenModule();
  /// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
  /// or nullptr on failure.
  mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, Module_ast& moduleAST);
  int dumpTolvaMLIR();
} // namespace toy
