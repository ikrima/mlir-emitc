#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "es2dsl/dialect/es2tolvaops.h.inl"
