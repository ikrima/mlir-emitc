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

enum class ExprASTKnd : uint8_t {
  VarDecl,
  Return,
  Num,
  Literal,
  Var,
  BinOp,
  Call,
  Print,
};

/// Base class for all expression nodes.
class Expr_ast {
public:

  Expr_ast(ExprASTKnd kind, SrcLoc_t location)
      : kind(kind), location(location) {}
  virtual ~Expr_ast() = default;

  ExprASTKnd getKind() const { return kind; }

  const SrcLoc_t &loc() { return location; }

private:
  const ExprASTKnd kind;
  SrcLoc_t location;
};

/// A block-list of expressions.
using ExprList_ast = std::vector<std::unique_ptr<Expr_ast>>;

/// Expression class for numeric literals like "1.0".
class NumExpr_ast : public Expr_ast {
  double Val;

public:
  NumExpr_ast(SrcLoc_t loc, double val) : Expr_ast(ExprASTKnd::Num, loc), Val(val) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == ExprASTKnd::Num; }
};

/// Expression class for a literal value.
class LitExpr_ast : public Expr_ast {
  std::vector<std::unique_ptr<Expr_ast>> values;
  std::vector<int64_t> dims;

public:
  LitExpr_ast(SrcLoc_t loc, std::vector<std::unique_ptr<Expr_ast>> values,
              std::vector<int64_t> dims)
      : Expr_ast(ExprASTKnd::Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<Expr_ast>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) {
    return c->getKind() == ExprASTKnd::Literal;
  }
};

/// Expression class for referencing a variable, like "a".
class VarExpr_ast : public Expr_ast {
  std::string name;

public:
  VarExpr_ast(SrcLoc_t loc, llvm::StringRef name)
      : Expr_ast(ExprASTKnd::Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == ExprASTKnd::Var; }
};

/// Expression class for defining a variable.
class VarDeclExpr_ast : public Expr_ast {
  std::string name;
  VarType type;
  std::unique_ptr<Expr_ast> initVal;

public:
  VarDeclExpr_ast(SrcLoc_t loc, llvm::StringRef name, VarType type,
                  std::unique_ptr<Expr_ast> initVal)
      : Expr_ast(ExprASTKnd::VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  Expr_ast *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) {
    return c->getKind() == ExprASTKnd::VarDecl;
  }
};


/// Expression class for function calls.
class CallExpr_ast : public Expr_ast {
  std::string callee;
  std::vector<std::unique_ptr<Expr_ast>> args;

public:
  CallExpr_ast(SrcLoc_t loc, const std::string &callee,
               std::vector<std::unique_ptr<Expr_ast>> args)
      : Expr_ast(ExprASTKnd::Call, loc), callee(callee), args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<Expr_ast>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == ExprASTKnd::Call; }
};


/// Expression class for a binary operator.
class BinaryExpr_ast : public Expr_ast {
  char op;
  std::unique_ptr<Expr_ast> lhs, rhs;

public:
  char getOp() { return op; }
  Expr_ast* getLHS() { return lhs.get(); }
  Expr_ast* getRHS() { return rhs.get(); }

  BinaryExpr_ast(SrcLoc_t loc, char Op, std::unique_ptr<Expr_ast> lhs,
    std::unique_ptr<Expr_ast> rhs)
    : Expr_ast(ExprASTKnd::BinOp, loc), op(Op), lhs(std::move(lhs)),
    rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const Expr_ast* c) { return c->getKind() == ExprASTKnd::BinOp; }
};

/// Expression class for a return operator.
class RetExpr_ast : public Expr_ast {
  llvm::Optional<std::unique_ptr<Expr_ast>> expr;

public:
  RetExpr_ast(SrcLoc_t loc, llvm::Optional<std::unique_ptr<Expr_ast>> expr)
    : Expr_ast(ExprASTKnd::Return, loc), expr(std::move(expr)) {}

  llvm::Optional<Expr_ast*> getExpr() {
    if (expr.hasValue())
      return expr->get();
    return llvm::None;
  }

  /// LLVM style RTTI
  static bool classof(const Expr_ast* c) { return c->getKind() == ExprASTKnd::Return; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public Expr_ast {
  std::unique_ptr<Expr_ast> arg;

public:
  PrintExprAST(SrcLoc_t loc, std::unique_ptr<Expr_ast> arg)
      : Expr_ast(ExprASTKnd::Print, loc), arg(std::move(arg)) {}

  Expr_ast *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const Expr_ast *c) { return c->getKind() == ExprASTKnd::Print; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
struct FnProto_ast {
  SrcLoc_t location;
  std::string name;
  std::vector<std::unique_ptr<VarExpr_ast>> args;

public:
  FnProto_ast(SrcLoc_t location, const std::string &name,
              std::vector<std::unique_ptr<VarExpr_ast>> args)
      : location(location), name(name), args(std::move(args)) {}

  const SrcLoc_t &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarExpr_ast>> getArgs() { return args; }
};

/// This class represents a function definition itself.
struct Func_ast {
  std::unique_ptr<FnProto_ast> proto;
  std::unique_ptr<ExprList_ast> body;

public:
  Func_ast(std::unique_ptr<FnProto_ast> proto,
           std::unique_ptr<ExprList_ast> body)
      : proto(std::move(proto)), body(std::move(body)) {}
  FnProto_ast *getProto() { return proto.get(); }
  ExprList_ast *getBody() { return body.get(); }
};

/// This class represents a list of functions to be processed together
struct Module_ast {
  std::vector<Func_ast> functions;

public:
  Module_ast(std::vector<Func_ast> functions)
      : functions(std::move(functions)) {}

  auto begin() -> decltype(functions.begin()) { return functions.begin(); }
  auto end() -> decltype(functions.end()) { return functions.end(); }
};

} // namespace es2
