#include "es2dsl/subsys/es2mlirsubsys.h"

#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"
#include "es2dsl/subsys/es2mlirsubsys.h"
#include "es2dsl/subsys/es2tlvast.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "../../../include/emitc/Target/Cpp.h"

#include "es2dsl/dialect/es2mlirpasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using namespace es2;
using namespace mlir::tolva;

namespace es2 {

struct TLVIRGenImpl {

  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp mlirMdl;

  /// The mlirOpbldr is a helper class to create IR inside a function. The
  /// mlirOpbldr is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  mlir::OpBuilder mlirOpbldr;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> mlirSymTbl;

  TLVIRGenImpl(mlir::MLIRContext &context) : mlirOpbldr(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(Module_ast &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    mlirMdl = mlir::ModuleOp::create(mlirOpbldr.getUnknownLoc());

    for (Func_ast &F : moduleAST) {
      auto func = mlirGen_func(F);
      if (!func)
        return nullptr;
      mlirMdl.push_back(func);
    }

    // Verify the module after we have finished constructing it, this will
    // check / the structural properties of the IR and invoke any specific
    // verifiers we / have on the Toy operations.
    if (failed(mlir::verify(mlirMdl))) {
      mlirMdl.emitError("module verification error");
      return nullptr;
    }

    return mlirMdl;
  }

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(SrcLoc_t loc) {
    return mlirOpbldr.getFileLineColLoc(mlirOpbldr.getIdentifier(*loc.file),
                                        loc.line, loc.col);
  }
  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (mlirSymTbl.count(var))
      return mlir::failure();
    mlirSymTbl.insert(var, value);
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(mlirOpbldr.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, mlirOpbldr.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const VarType &type) { return getType(type.shape); }

#if 1
  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::FuncOp mlirGen_fnproto(FnProto_ast &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                               getType(VarType{}));
    auto func_type = mlirOpbldr.getFunctionType(arg_types, llvm::None);
    return mlir::FuncOp::create(location, proto.getName(), func_type);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen_func(Func_ast &_fnast) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(mlirSymTbl);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(mlirGen_fnproto(*_fnast.getProto()));
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
    auto protoArgs = _fnast.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto &name_value :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (mlir::failed(declare(std::get<0>(name_value)->getName(),
                               std::get<1>(name_value))))
        return nullptr;
    }

    // Set the insertion point in the mlirOpbldr to the beginning of the
    // function body, it will be used throughout the codegen to create
    // operations in this function.
    mlirOpbldr.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen_exprlst(*_fnast.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      mlirOpbldr.create<ReturnOp>(loc(_fnast.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(mlirOpbldr.getFunctionType(
          function.getType().getInputs(), getType(VarType{})));
    }

    return function;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen_exprlst(ExprList_ast &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> var_scope(mlirSymTbl);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExpr_ast>(expr.get())) {
        if (!mlirGen_vardecl(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<RetExpr_ast>(expr.get()))
        return mlirGen_ret(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen_print(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen_expr(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult mlirGen_print(PrintExprAST &call) {
    auto arg = mlirGen_expr(*call.getArg());
    if (!arg)
      return mlir::failure();

    mlirOpbldr.create<PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen_num(NumExpr_ast &num) {
    return mlirOpbldr.create<ConstantOp>(loc(num.loc()), num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen_expr(Expr_ast &expr) {
    switch (expr.getKind()) {
    case ExprASTKnd::BinOp:
      return mlirGen_binop(cast<BinaryExpr_ast>(expr));
    case ExprASTKnd::Var:
      return mlirGen_var(cast<VarExpr_ast>(expr));
    case ExprASTKnd::Literal:
      return mlirGen_lit(cast<LitExpr_ast>(expr));
    case ExprASTKnd::Call:
      return mlirGen_call(cast<CallExpr_ast>(expr));
    case ExprASTKnd::Num:
      return mlirGen_num(cast<NumExpr_ast>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine((uint8_t)expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen_ret(RetExpr_ast &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().hasValue()) {
      if (!(expr = mlirGen_expr(*ret.getExpr().getValue())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    mlirOpbldr.create<ReturnOp>(location, expr ? makeArrayRef(expr)
                                               : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen_lit(LitExpr_ast &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = mlirOpbldr.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return mlirOpbldr.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(Expr_ast &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LitExpr_ast>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumExpr_ast>(expr) && "expected literal or number expr");
    data.push_back(cast<NumExpr_ast>(expr).getValue());
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen_vardecl(VarDeclExpr_ast &vardecl) {
    auto init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen_expr(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = mlirOpbldr.create<ReshapeOp>(loc(vardecl.loc()),
                                           getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Emit a binary operation
  mlir::Value mlirGen_binop(BinaryExpr_ast &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen_expr(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen_expr(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return mlirOpbldr.create<AddOp>(location, lhs, rhs);
    case '*':
      return mlirOpbldr.create<MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen_var(VarExpr_ast &expr) {
    if (auto variable = mlirSymTbl.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen_call(CallExpr_ast &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen_expr(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return mlirOpbldr.create<TransposeOp>(location, operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return mlirOpbldr.create<GenericCallOp>(location, callee, operands);
  }

#endif
};

} // namespace es2

/// Build full Module. A module is a list of function definitions.
std::unique_ptr<Module_ast> es2::astGenModule() {
  using namespace std;

  shared_ptr<string> file = make_shared<string>(
      "D:/ikrima/src/personal/tolva/code/mlir-emitc/es2dsl/test/dummy.tolva");
  shared_ptr<string> file_content = make_shared<string>(
      R"(
def multiply_transpose(a, b) {
var a2 = transpose(a);
var b2 = transpose(b);
var ret = a2 * b2;
return ret;
}
)");

  vector<Func_ast> mdlfns;
  {
    vector<unique_ptr<VarExpr_ast>> args;
    args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 2, 24}, "a"));
    args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 2, 27}, "b"));
    unique_ptr<FnProto_ast> fnproto = make_unique<FnProto_ast>(
        SrcLoc_t{file, 2, 1}, "multiply_transpose", move(args));

    unique_ptr<ExprList_ast> fnbody = make_unique<ExprList_ast>();
    {
      {
        vector<unique_ptr<Expr_ast>> args;
        args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 3, 20}, "a"));
        fnbody->emplace_back(make_unique<VarDeclExpr_ast>(
            SrcLoc_t{file, 3, 1}, "a2", VarType{},
            make_unique<CallExpr_ast>(SrcLoc_t{file, 3, 10}, "transpose",
                                      move(args))));
      }
      {
        vector<unique_ptr<Expr_ast>> args;
        args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 4, 20}, "b"));
        fnbody->emplace_back(make_unique<VarDeclExpr_ast>(
            SrcLoc_t{file, 4, 1}, "b2", VarType{},
            make_unique<CallExpr_ast>(SrcLoc_t{file, 4, 10}, "transpose",
                                      move(args))));
      }
      {
        fnbody->emplace_back(make_unique<VarDeclExpr_ast>(
            SrcLoc_t{file, 5, 1}, "ret", VarType{},
            make_unique<BinaryExpr_ast>(
                SrcLoc_t{file, 5, 16}, '*',
                make_unique<VarExpr_ast>(SrcLoc_t{file, 5, 11}, "a2"),
                make_unique<VarExpr_ast>(SrcLoc_t{file, 5, 16}, "b2"))));
      }
      {
        ;
        fnbody->emplace_back(make_unique<RetExpr_ast>(
            SrcLoc_t{file, 6, 1},
            llvm::Optional<unique_ptr<Expr_ast>>(
                make_unique<VarExpr_ast>(SrcLoc_t{file, 6, 8}, "ret"))));
      }
    }

    mdlfns.emplace_back(move(fnproto), move(fnbody));
  }

  return make_unique<Module_ast>(move(mdlfns));
}

mlir::OwningModuleRef es2::mlirGen(mlir::MLIRContext &context,
                                   Module_ast &moduleAST) {
  return TLVIRGenImpl(context).mlirGen(moduleAST);
}

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningModuleRef &module) {
  using namespace std;
  unique_ptr<Module_ast> tlvmdlast = es2::astGenModule();
  if (!tlvmdlast)
    return 6;
  module = mlirGen(context, *tlvmdlast);
  return !module ? 1 : 0;

  //// Handle '.toy' input to the compiler.
  // if (inputType != InputType::MLIR &&
  //  !llvm::StringRef(inputFilename).endswith(".mlir")) {
  //  auto moduleAST = parseInputFile(inputFilename);
  //  if (!moduleAST)
  //    return 6;
  //  module = mlirGen(context, *moduleAST);
  //  return !module ? 1 : 0;
  //}

  //// Otherwise, the input is '.mlir'.
  // llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
  //  llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  // if (std::error_code EC = fileOrErr.getError()) {
  //  llvm::errs() << "Could not open input file: " << EC.message() << "\n";
  //  return -1;
  //}

  //// Parse the input mlir.
  // sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  // module = mlir::parseSourceFile(sourceMgr, &context);
  // if (!module) {
  //  llvm::errs() << "Error can't load file " << inputFilename << "\n";
  //  return 3;
  //}
  return 0;
}

int es2::dumpTolvaMLIR() {
  using namespace std;
  mlir::MLIRContext context(/*loadAllDialects=*/false);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::tolva::TolvaDialect>();
  mlir::OwningModuleRef module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  // clang-format off
  llvm::outs() << "//------------------------------------------------------------------------------//\n";
  llvm::outs() << "TLVIR:\n";
  llvm::outs() << "//------------------------------------------------------------------------------//\n";
  module->dump();
  llvm::outs() << "\n\n";
  llvm::outs() << "//------------------------------------------------------------------------------//\n\n\n";
  // clang-format on

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  mlirTranslate(*module->getOperation());
#if 0
  {

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToCpp = true;

    if (isLoweringToCpp) {
      // Finish lowering the toy IR to the LLVM dialect.
      pm.addPass(mlir::tolva::createLowerToCppPass());
    }

    if (mlir::failed(pm.run(*module)))
      return 4;

    module->dump();
    return 0;
  }
#endif // 0

  return 0;
}

int es2::mlirTranslate(mlir::Operation &op) {
  return failed(mlir::emitc::TranslateToCpp(op, llvm::outs(), false));
}