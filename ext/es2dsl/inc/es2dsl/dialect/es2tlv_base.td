//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOLVA_DIALECT_TLV_BASE
#define TOLVA_DIALECT_TLV_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tolva base dialect used for types common across subdialects.
//===----------------------------------------------------------------------===//

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Tolva_Dialect : Dialect {
  let name = "tolva";
  let cppNamespace = "::mlir::tolva";
  let summary = [{
    A dialect used for types common across Tolva subdialects.
  }];
}

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}


#endif  // TOLVA_DIALECT_TOLVA_BASE
