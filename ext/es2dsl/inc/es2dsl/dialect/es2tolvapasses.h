#pragma once

#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace llvm {
class raw_ostream;
}

namespace mlir {
class Pass;
class Operation;

namespace tolva {
  std::unique_ptr<mlir::Pass> createShapeInferencePass();

  /// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
  /// for a subset of the Toy IR (e.g. matmul).
  std::unique_ptr<mlir::Pass> createLowerToAffinePass();

  /// Create a pass for lowering operations the remaining `Toy` operations, as
  /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
  std::unique_ptr<mlir::Pass> createLowerToCppPass();


  /// Translates the given operation to C++ code. The operation or operations in
  /// the region of 'op' need almost all be in EmitC dialect.
  mlir::LogicalResult TranslateToCpp(mlir::Operation& op, llvm::raw_ostream& os, bool trailingSemicolon = false);

  /// Create a pass for lowering operations the remaining `Toy` operations, as
  /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
  std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
}    // namespace tolva

}    // namespace mlir
