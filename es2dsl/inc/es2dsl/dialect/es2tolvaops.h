#pragma once

#include "es2dsl/dialect/es2tolvadialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "es2dsl/dialect/es2tolvaops.h.inl"
