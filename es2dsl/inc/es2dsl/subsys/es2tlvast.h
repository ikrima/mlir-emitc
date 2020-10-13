//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace es2 {

/// A variable type with shape information.
struct VarType {
  std::vector<int64_t> shape;
};

/// Structure definition a location in a file.
struct SrcLoc_t {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

/// Base class for all expression nodes.
class Expr_ast {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  Expr_ast(ExprASTKind kind, SrcLoc_t location)
      : kind(kind), location(location) {}
  virtual ~Expr_ast() = default;

  ExprASTKind getKind() const { return kind; }

  const SrcLoc_t &loc() { return location; }

private:
  const ExprASTKind kind;
  SrcLoc_t location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<Expr_ast>>;

/// Expression class for numeric literals like "1.0".
class NumExpr_ast : public Expr_ast {
  double Val;

public:
  NumExpr_ast(SrcLoc_t loc, double val) : Expr_ast(Expr_Num, loc), Val(val) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LitExpr_ast : public Expr_ast {
  std::vector<std::unique_ptr<Expr_ast>> values;
  std::vector<int64_t> dims;

public:
  LitExpr_ast(SrcLoc_t loc, std::vector<std::unique_ptr<Expr_ast>> values,
              std::vector<int64_t> dims)
      : Expr_ast(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<Expr_ast>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) {
    return c->getKind() == Expr_Literal;
  }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public Expr_ast {
  std::string name;

public:
  VariableExprAST(SrcLoc_t loc, llvm::StringRef name)
      : Expr_ast(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExpr_ast : public Expr_ast {
  std::string name;
  VarType type;
  std::unique_ptr<Expr_ast> initVal;

public:
  VarDeclExpr_ast(SrcLoc_t loc, llvm::StringRef name, VarType type,
                  std::unique_ptr<Expr_ast> initVal)
      : Expr_ast(Expr_VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  Expr_ast *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) {
    return c->getKind() == Expr_VarDecl;
  }
};

/// Expression class for function calls.
class CallExpr_ast : public Expr_ast {
  std::string callee;
  std::vector<std::unique_ptr<Expr_ast>> args;

public:
  CallExpr_ast(SrcLoc_t loc, const std::string &callee,
               std::vector<std::unique_ptr<Expr_ast>> args)
      : Expr_ast(Expr_Call, loc), callee(callee), args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<Expr_ast>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public Expr_ast {
  std::unique_ptr<Expr_ast> arg;

public:
  PrintExprAST(SrcLoc_t loc, std::unique_ptr<Expr_ast> arg)
      : Expr_ast(Expr_Print, loc), arg(std::move(arg)) {}

  Expr_ast *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == Expr_Print; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
struct FnProto_ast {
  SrcLoc_t location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  FnProto_ast(SrcLoc_t location, const std::string &name,
              std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(location), name(name), args(std::move(args)) {}

  const SrcLoc_t &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
};

/// This class represents a function definition itself.
struct Func_ast {
  std::unique_ptr<FnProto_ast> proto;
  std::unique_ptr<ExprASTList> body;

public:
  Func_ast(std::unique_ptr<FnProto_ast> proto,
           std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}
  FnProto_ast *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }
};

/// This class represents a list of functions to be processed together
struct TlvModule_ast {
  std::vector<Func_ast> functions;

public:
  TlvModule_ast(std::vector<Func_ast> functions)
      : functions(std::move(functions)) {}

  auto begin() -> decltype(functions.begin()) { return functions.begin(); }
  auto end() -> decltype(functions.end()) { return functions.end(); }
};

} // namespace es2
