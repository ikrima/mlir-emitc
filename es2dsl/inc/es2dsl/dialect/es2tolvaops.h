#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace es2 {

class ConstantOp : public mlir::Op<ConstantOp,
                                   /// The ConstantOp takes no inputs.
                                   mlir::OpTrait::ZeroOperands,
                                   /// The ConstantOp returns a single result.
                                   mlir::OpTrait::OneResult> {

public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations can provide additional verification beyond the traits they
  /// define. Here we will ensure that the specific invariants of the constant
  /// operation are upheld, for example the result type must be of TensorType.
  LogicalResult verify();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the builder to allow for easily generating
  /// instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};

} // namespace es2
} // namespace mlir