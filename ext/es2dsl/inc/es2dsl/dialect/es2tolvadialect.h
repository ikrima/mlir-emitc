#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir { namespace tolva {

/// Include the auto-generated definitions for the shape inference interfaces.
#include "es2dsl/tblgen/es2tlv_base.h.inl"

  /// This is the definition of the Toy dialect. A dialect inherits from
  /// mlir::Dialect and registers custom attributes, operations, and types (in its
  /// constructor). It can also override some general behavior exposed via virtual
  /// methods.
  class TolvaDialect : public mlir::Dialect
  {
  public:
    explicit TolvaDialect(mlir::MLIRContext* ctx);

    /// Provide a utility accessor to the dialect namespace. This is used by
    /// several utilities for casting between dialects.
    static llvm::StringRef getDialectNamespace() { return "tolva"; }
  };

}}    // namespace mlir::tolva
