//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOLVA_DIALECT_TOLVA_BASE
#define TOLVA_DIALECT_TOLVA_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tolva base dialect used for types common across subdialects.
//===----------------------------------------------------------------------===//

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Tolva_Dialect : Dialect {
  let name = "tolva";
  let cppNamespace = "::mlir::tolva";
}


#endif  // TOLVA_DIALECT_TOLVA_BASE
