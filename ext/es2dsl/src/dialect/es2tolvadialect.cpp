
#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::tolva;

//===----------------------------------------------------------------------===//
// TolvaDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
TolvaDialect::TolvaDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<TolvaDialect>()) {

#define GET_OP_LIST
  addOperations<
#include "es2dsl/tblgen/es2tlv_ops.cpp.inl"
  >();
}