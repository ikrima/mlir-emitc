#ifndef TOLVA_DIALECT_TOLVA_OPS
#define TOLVA_DIALECT_TOLVA_OPS

include "es2dsl/dialect/es2tolvabase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Tolva_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Tolva_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Executable ABI
//===----------------------------------------------------------------------===//
def ConstantOp : Tolva_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}

#endif  // TOLVA_DIALECT_TOLVA_OPS
