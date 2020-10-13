#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace tolva {
/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToCppPass();
} // namespace es2

} // namespace mlir
