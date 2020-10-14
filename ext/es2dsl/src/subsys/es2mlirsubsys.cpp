#include "es2dsl/subsys/es2mlirsubsys.h"

#include "es2dsl/dialect/es2mlirpasses.h"
#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"
#include "es2dsl/subsys/es2mlirsubsys.h"
#include "es2dsl/subsys/es2tlvast.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/EDSC/Builders.h"
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

/// Build full Module. A module is a list of function definitions.
std::unique_ptr<Module_ast> es2::astGenMdl_multiply_transpose() {
  using namespace std;

  shared_ptr<string> file         = make_shared<string>("dummy.tolva");
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
    unique_ptr<FnProto_ast> fnproto = make_unique<FnProto_ast>(SrcLoc_t{file, 2, 1}, "multiply_transpose", move(args));

    unique_ptr<ExprList_ast> fnbody = make_unique<ExprList_ast>();
    {
      {
        vector<unique_ptr<Expr_ast>> args;
        args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 3, 20}, "a"));
        fnbody->emplace_back(make_unique<VarDeclExpr_ast>(
          SrcLoc_t{file, 3, 1}, "a2", VarType{}, make_unique<CallExpr_ast>(SrcLoc_t{file, 3, 10}, "transpose", move(args))));
      }
      {
        vector<unique_ptr<Expr_ast>> args;
        args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 4, 20}, "b"));
        fnbody->emplace_back(make_unique<VarDeclExpr_ast>(
          SrcLoc_t{file, 4, 1}, "b2", VarType{}, make_unique<CallExpr_ast>(SrcLoc_t{file, 4, 10}, "transpose", move(args))));
      }
      {
        fnbody->emplace_back(make_unique<VarDeclExpr_ast>(SrcLoc_t{file, 5, 1}, "ret", VarType{},
          make_unique<BinaryExpr_ast>(SrcLoc_t{file, 5, 16}, '*', make_unique<VarExpr_ast>(SrcLoc_t{file, 5, 11}, "a2"),
            make_unique<VarExpr_ast>(SrcLoc_t{file, 5, 16}, "b2"))));
      }
      {
        ;
        fnbody->emplace_back(make_unique<RetExpr_ast>(
          SrcLoc_t{file, 6, 1}, llvm::Optional<unique_ptr<Expr_ast>>(make_unique<VarExpr_ast>(SrcLoc_t{file, 6, 8}, "ret"))));
      }
    }

    mdlfns.emplace_back(move(fnproto), move(fnbody));
  }

  return make_unique<Module_ast>(move(mdlfns));
}

std::unique_ptr<Module_ast> es2::astGenMdl_transpose_transpose() {
  using namespace std;

  shared_ptr<string> file         = make_shared<string>("dummy.tolva");
  shared_ptr<string> file_content = make_shared<string>(
    R"(
def transpose_transpose(x) {
  return transpose(transpose(x));
}
)");

  vector<Func_ast> mdlfns;
  {
    vector<unique_ptr<VarExpr_ast>> args;
    args.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 2, 24}, "x"));
    unique_ptr<FnProto_ast> fnproto = make_unique<FnProto_ast>(SrcLoc_t{file, 2, 1}, "transpose_transpose", move(args));

    unique_ptr<ExprList_ast> fnbody = make_unique<ExprList_ast>();
    {
      vector<unique_ptr<Expr_ast>> args1;
      args1.emplace_back(make_unique<VarExpr_ast>(SrcLoc_t{file, 3, 20}, "x"));
      unique_ptr<CallExpr_ast> call_innertrans = make_unique<CallExpr_ast>(SrcLoc_t{file, 3, 10}, "transpose", move(args1));

      vector<unique_ptr<Expr_ast>> args2;
      args2.emplace_back(move(call_innertrans));
      unique_ptr<CallExpr_ast> call_outertrans = make_unique<CallExpr_ast>(SrcLoc_t{file, 3, 10}, "transpose", move(args2));

      fnbody->emplace_back(make_unique<RetExpr_ast>(SrcLoc_t{file, 6, 1}, llvm::Optional<unique_ptr<Expr_ast>>(move(call_outertrans))));
    }

    mdlfns.emplace_back(move(fnproto), move(fnbody));
  }

  return make_unique<Module_ast>(move(mdlfns));
}


mlir::OwningModuleRef mlirGenMdl_multiply_transpose(mlir::MLIRContext& _mlirctx) {
  using namespace std;

  llvm::StringRef    file         = "dummy.tolva";
  shared_ptr<string> file_content = make_shared<string>(
    R"(
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
)");

  using llvm::makeArrayRef;
  using mlir::DenseElementsAttr;
  using mlir::FileLineColLoc;
  using mlir::FloatType;
  using mlir::FuncOp;
  using mlir::ModuleOp;
  using mlir::RankedTensorType;
  using mlir::UnknownLoc;
  using mlir::UnrankedTensorType;
  using mlir::edsc::OperationBuilder;
  using mlir::edsc::ScopedContext;
  using mlir::edsc::ValueBuilder;
  ModuleOp mlirMdl = ModuleOp::create(UnknownLoc::get(&_mlirctx));

  // clang-format off
  // multiply_transpose
  {
    FuncOp f = FuncOp::create(FileLineColLoc::get(file, 2, 1, &_mlirctx), "multiply_transpose",
      mlir::FunctionType::get(
        {UnrankedTensorType::get(FloatType::getF64(&_mlirctx)), UnrankedTensorType::get(FloatType::getF64(&_mlirctx))},
        {UnrankedTensorType::get(FloatType::getF64(&_mlirctx))}, &_mlirctx));
    f.addEntryBlock();
    {
      mlir::OpBuilder builder(f.getBody());
      ScopedContext   scope(builder, f.getLoc());
      ReturnOp        ret = OperationBuilder<ReturnOp>(makeArrayRef<mlir::Value>(
        ValueBuilder<MulOp>(ValueBuilder<TransposeOp>(f.getArgument(0)), ValueBuilder<TransposeOp>(f.getArgument(1)))));
    }
    f.setVisibility(mlir::FuncOp::Visibility::Private);
    mlirMdl.push_back(f);
  }

  // main
  {
    FuncOp f = FuncOp::create(FileLineColLoc::get(file, 2, 1, &_mlirctx), "main", mlir::FunctionType::get({}, {}, &_mlirctx));
    f.addEntryBlock();
    {
      mlir::OpBuilder  builder(f.getBody());
      ScopedContext    scope(builder, f.getLoc());
      mlir::Type       elementType = FloatType::getF64(&_mlirctx);
      RankedTensorType dataType    = RankedTensorType::get({2, 3}, FloatType::getF64(&_mlirctx));

      mlir::Value vara = ValueBuilder<ReshapeOp>(
        dataType, ValueBuilder<ConstantOp>(dataType, DenseElementsAttr::get(dataType, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
      mlir::Value varb = ValueBuilder<ReshapeOp>(
        dataType, ValueBuilder<ConstantOp>(dataType, DenseElementsAttr::get(dataType, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
      mlir::Value varc = ValueBuilder<GenericCallOp>("multiply_transpose", makeArrayRef<mlir::Value>({vara, varb}));
      mlir::Value vard = ValueBuilder<GenericCallOp>("multiply_transpose", makeArrayRef<mlir::Value>({varb, vara}));
      
      PrintOp printop = OperationBuilder<PrintOp>(vard);

      OperationBuilder<ReturnOp>();
    }
    mlirMdl.push_back(f);
  }

  // clang-format on

  // Verify the module after we have finished constructing it, this will
  // check / the structural properties of the IR and invoke any specific
  // verifiers we / have on the Toy operations.
  if (failed(mlir::verify(mlirMdl))) {
    mlirMdl.emitError("module verification error");
    return nullptr;
  }

  return mlirMdl;
}

void dumpTolvaMLIRPass(mlir::OwningModuleRef& _mlirMdl, const char* _passname) {
  // clang-format off
  llvm::outs() << "//------------------------------------------------------------------------------//\n";
  llvm::outs() << _passname << "\n";
  llvm::outs() << "//------------------------------------------------------------------------------//\n";
  _mlirMdl->dump();
  llvm::outs() << "\n\n";
  llvm::outs() << "//------------------------------------------------------------------------------//\n\n\n";
  // clang-format on
}


int loadTolvaMLIR(llvm::SourceMgr& sourceMgr, mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
  using namespace std;
  if (false) {
    unique_ptr<Module_ast> tlvmdlast = es2::astGenMdl_transpose_transpose();
    if (!tlvmdlast) return 6;
    module = mlirGen(context, *tlvmdlast);
  }
  else {
    module = mlirGenMdl_multiply_transpose(context);
  }
  return !module ? 1 : 0;
}


int es2::genTolvaMLIR() {
  using namespace std;
  mlir::MLIRContext context(/*loadAllDialects=*/false);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::tolva::TolvaDialect>();
  mlir::OwningModuleRef            module;
  llvm::SourceMgr                  sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadTolvaMLIR(sourceMgr, context, module)) return error;

  dumpTolvaMLIRPass(module, "TLVIR (MLIR):");
  const bool bCanonicalizationOnly = false;
  const bool bEnableOpt            = true;
  const bool bEnableLowering       = false;

  {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Canonicalization only
    if (bCanonicalizationOnly) {
      pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
      if (mlir::failed(pm.run(*module))) {
        return 4;
      }
      dumpTolvaMLIRPass(module, "TLVIR (Canonical):");
      return 0;
    }

    // Optimization passes
    if (bEnableOpt || bEnableLowering) {
      // Inline all functions into main and then delete them.
      pm.addPass(mlir::createInlinerPass());

      // Now that there is only one function, we can infer the shapes of each of
      // the operations.
      mlir::OpPassManager& optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::tolva::createShapeInferencePass());
      optPM.addPass(mlir::createCanonicalizerPass());    // Add a run of the canonicalizer to optimize the mlir module.
      optPM.addPass(mlir::createCSEPass());
    }

    if (bEnableLowering) {
      // Partially lower the toy dialect with a few cleanups afterwards.
      pm.addPass(mlir::tolva::createLowerToAffinePass());

      mlir::OpPassManager& optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      // Add optimizations if enabled.
      if (bEnableOpt) {
        optPM.addPass(mlir::createLoopFusionPass());
        optPM.addPass(mlir::createMemRefDataFlowOptPass());
      }
    }

    if (mlir::failed(pm.run(*module))) {
      return 4;
    }
    dumpTolvaMLIRPass(module, "TLVIR (Opt):");
  }

#if 0
  failed(mlir::emitc::TranslateToCpp(*module->getOperation(), llvm::outs(), false));
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
#endif    // 0

  return 0;
}