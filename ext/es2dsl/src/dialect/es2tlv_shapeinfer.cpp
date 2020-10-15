#include "es2dsl/dialect/es2tolvapasses.h"
#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace tolva;


/// Include the auto-generated definitions for the shape inference interfaces.
#include "es2dsl/tblgen/es2tlv_base.cpp.inl"


namespace {
/// The ShapeInferencePass is a FunctionPass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
class ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass, FunctionPass>
{
public:
  void runOnFunction() override {
    auto f = getFunction();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation*, 16> opWorklist;
    f.walk([&](mlir::Operation* op) {
      if (returnsDynamicShape(op)) opWorklist.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end()) break;

      Operation* op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      llvm::dbgs() << "Inferring shape for: " << *op << "\n";
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      }
      else {
        op->emitError(
          "unable to infer shape of operation without shape "
          "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ") << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(Operation* op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) { return operandType.isa<RankedTensorType>(); });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returnsDynamicShape(Operation* op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) { return !resultType.isa<RankedTensorType>(); });
  }
};
}    // end anonymous namespace

//===----------------------------------------------------------------------===//
/// Create a Shape Inference pass
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> mlir::tolva::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
