//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tolva;


//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
  double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser& parser,
  mlir::OperationState& result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
    parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter& printer, ConstantOp op) {
  printer << "toy.constant ";
  printer.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{ "value" });
  printer << op.value();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
      "return type must match the one of the attached value "
      "attribute: ")
      << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
        "return type shape mismatches its attribute at dimension ")
        << dim << ": " << attrType.getShape()[dim]
        << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "es2dsl/dialect/es2tolvaops.cpp.inl"
