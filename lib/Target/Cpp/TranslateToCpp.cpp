//===- TranslateToCpp.cpp - Translating to C++ calls ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Target/Cpp.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <stack>

#define DEBUG_TYPE "translate-to-cpp"

using namespace mlir;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STL as functions used on
/// each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end)
    return success();
  if (failed(each_fn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    if (failed(each_fn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor each_fn,
                                         NullaryFunctor between_fn) {
  return interleaveWithError(c.begin(), c.end(), each_fn, between_fn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor each_fn) {
  return interleaveWithError(c.begin(), c.end(), each_fn,
                             [&]() { os << ", "; });
}

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  explicit CppEmitter(raw_ostream &os);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon = true);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  LogicalResult emitTypes(ArrayRef<Type> types);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CppEmitter &emitter) : mapperScope(emitter.mapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
    }
    ~Scope() { emitter.valueInScopeCount.pop(); }

  private:
    llvm::ScopedHashTableScope<Value, std::string> mapperScope;
    CppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  /// Returns the output stream.
  raw_ostream &ostream() { return os; };

private:
  using ValMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_ostream &os;

  /// Map from value to name of C++ variable that contain the name.
  ValMapper mapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
};
} // namespace

static LogicalResult printConstantOp(CppEmitter &emitter,
                                     ConstantOp constantOp) {
  auto &os = emitter.ostream();
  emitter.emitType(constantOp.getType());
  os << " " << emitter.getOrCreateName(constantOp.getResult()) << '{';
  if (failed(emitter.emitAttribute(constantOp.getValue())))
    return constantOp.emitError("unable to emit constant value");
  os << '}';
  return success();
}

static LogicalResult printCallOp(CppEmitter &emitter, CallOp callOp) {
  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  auto &os = emitter.ostream();
  os << callOp.getCallee() << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printCallOp(CppEmitter &emitter, emitc::CallOp callOp) {
  auto &os = emitter.ostream();
  auto &op = *callOp.getOperation();
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << callOp.callee() << "(";

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = attr.dyn_cast<IntegerAttr>()) {
      // Index attributes are treated specially as operand index.
      if (t.getType().isIndex()) {
        auto idx = t.getInt();
        if ((idx < 0) || (idx >= op.getNumOperands()))
          return op.emitOpError() << "invalid operand index";
        if (!emitter.hasValueInScope(op.getOperand(idx)))
          return op.emitOpError()
                 << "operand " << idx << "'s value not defined in scope";
        os << emitter.getOrCreateName(op.getOperand(idx));
        return success();
      }
    }
    return emitter.emitAttribute(attr);
  };

  //if (callOp.argsAttr()) {
  //  callOp.dump();
  //}
  auto emittedArgs =
      callOp.args() ? interleaveCommaWithError(*callOp.args(), os, emitArgs)
                    : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printReturnOp(CppEmitter &emitter, ReturnOp returnOp) {
  auto &os = emitter.ostream();
  os << "return ";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << "std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printModule(CppEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  os << "// Forward declare functions.\n";
  for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
    if (failed(emitter.emitTypes(funcOp.getType().getResults())))
      return funcOp.emitError() << "failed to convert operand type";
    os << " " << funcOp.getName() << "(";
    if (failed(interleaveCommaWithError(
            funcOp.getArguments(), os, [&](BlockArgument arg) {
              return emitter.emitType(arg.getType());
            })))
      return failure();
    os << ");\n";
  }
  os << "\n";

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemiColon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printFunction(CppEmitter &emitter, FuncOp functionOp) {
  auto &blocks = functionOp.getBlocks();
  if (blocks.size() != 1)
    return functionOp.emitOpError() << "only single block functions supported";

  CppEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  if (failed(emitter.emitTypes(functionOp.getType().getResults())))
    return functionOp.emitError() << "unable to emit all types";
  os << " " << functionOp.getName();

  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(arg.getType())))
              return functionOp.emitError() << "unable to emit arg "
                                            << arg.getArgNumber() << "'s type";
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";

  for (Operation &op : functionOp.front()) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  os << "}\n";
  return success();
}

CppEmitter::CppEmitter(raw_ostream &os) : os(os) { valueInScopeCount.push(0); }

/// Return the existing or a new name for a Value*.
StringRef CppEmitter::getOrCreateName(Value val) {
  if (!mapper.count(val))
    mapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *mapper.begin(val);
}

bool CppEmitter::hasValueInScope(Value val) { return mapper.count(val); }

LogicalResult CppEmitter::emitAttribute(Attribute attr) {
  if (auto iAttr = attr.dyn_cast<IntegerAttr>()) {
    os << iAttr.getValue();
    return success();
  }
  if (auto dense = attr.dyn_cast<DenseIntElementsAttr>()) {
    os << '{';
    interleaveComma(dense.getIntValues(), os);
    os << '}';
    return success();
  }
  return failure();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (auto attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.first.strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.first.strref()))
      return success();
    os << "/* " << attr.first << " */";
    if (failed(emitAttribute(attr.second)))
      return op.emitError() << "unable to emit attribute " << attr.second;
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    auto result = op.getResult(0);
    if (failed(emitType(result.getType())))
      return failure();
    os << " " << getOrCreateName(result) << " = ";
    break;
  }
  default:
    for (auto result : op.getResults()) {
      if (failed(emitType(result.getType())))
        return failure();
      os << " " << getOrCreateName(result) << ";\n";
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, Operation &op) {
  if (auto callOp = dyn_cast<CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto callOp = dyn_cast<emitc::CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto constantOp = dyn_cast<ConstantOp>(op))
    return printConstantOp(emitter, constantOp);
  if (auto returnOp = dyn_cast<ReturnOp>(op))
    return printReturnOp(emitter, returnOp);
  if (auto moduleOp = dyn_cast<ModuleOp>(op))
    return printModule(emitter, moduleOp);
  if (auto funcOp = dyn_cast<FuncOp>(op))
    return printFunction(emitter, funcOp);
  if (isa<ModuleTerminatorOp>(op))
    return success();

  return op.emitOpError() << "unable to find printer for op";
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  if (failed(printOperation(*this, op)))
    return failure();
  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitType(Type type) {
  if (auto itype = type.dyn_cast<IntegerType>()) {
    switch (itype.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 32:
      return (os << "int32_t"), success();
    case 64:
      return (os << "int64_t"), success();
    default:
      return failure();
    }
  }
  // TODO: Change to be EmitC specific.
  if (auto ot = type.dyn_cast<OpaqueType>()) {
    os << ot.getTypeData();
    return success();
  }
  return failure();
}

LogicalResult CppEmitter::emitTypes(ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(types.front());
  default:
    os << "std::tuple<";
    if (failed(interleaveCommaWithError(
            types, os, [&](Type type) { return emitType(type); })))
      return failure();
    os << ">";
    return success();
  }
}

LogicalResult emitc::TranslateToCpp(Operation &op, raw_ostream &os,
                                    bool trailingSemicolon) {
  CppEmitter emitter(os);
  return emitter.emitOperation(op, trailingSemicolon);
}
