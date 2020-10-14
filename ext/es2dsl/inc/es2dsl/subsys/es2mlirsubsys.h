//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
class Operation;
}    // namespace mlir
namespace llvm {
class SourceMgr;
}

namespace es2 {
struct Module_ast;

struct DSLSubsys_api {
  bool bCanonicalizationOnly = false;
  bool bOptimize             = false;
  bool bLowerToAffine        = true;
  bool bLowerToLLVM          = false;
  bool bDumpLLVMIR           = false;
  bool bRunJIT               = false;

  /// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
  /// or nullptr on failure.
  mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, Module_ast& moduleAST);
  int                   loadTolvaMLIR(llvm::SourceMgr& sourceMgr, mlir::MLIRContext& context, mlir::OwningModuleRef& module);
  int                   dumpLLVMIR(mlir::OwningModuleRef& module);
  int                   runJit(mlir::OwningModuleRef& module);
  int                   genTolvaMLIR();
};

}    // namespace es2
