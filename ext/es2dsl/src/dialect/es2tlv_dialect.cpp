
#include "es2dsl/dialect/es2tolvadialect.h"

#include "es2dsl/dialect/es2tolvaops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tolva;

//===----------------------------------------------------------------------===//
// TolvaInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct TolvaInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within toy can be inlined.
  bool isLegalToInline(Operation*, Region*, BlockAndValueMapping&) const final { return true; }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation* op, ArrayRef<Value> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto& it : llvm::enumerate(returnOp.getOperands())) valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation* materializeCallConversion(OpBuilder& builder, Value input,
    Type resultType,
    Location conversionLoc) const final {
    return builder.create<mlir::tolva::CastOp>(conversionLoc, resultType, input);
  }
};


//===----------------------------------------------------------------------===//
// TolvaDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
TolvaDialect::TolvaDialect(mlir::MLIRContext* ctx)
  : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<TolvaDialect>()) {
  addInterfaces<TolvaInlinerInterface>();

#define GET_OP_LIST
  addOperations<
#include "es2dsl/tblgen/es2tlv_ops.cpp.inl"
    >();
}