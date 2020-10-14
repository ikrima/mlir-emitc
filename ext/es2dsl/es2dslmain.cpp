#include "es2dsl/subsys/es2mlirsubsys.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"

#include <memory>

using namespace es2;

int main(int argc, char** argv) {
  mlir::registerAllDialects();

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  DSLSubsys_api dslsubsys;
  dslsubsys.bCanonicalizationOnly = false;
  dslsubsys.bOptimize             = true;
  dslsubsys.bLowerToAffine        = true;
  dslsubsys.bLowerToLLVM          = true;
  dslsubsys.bDumpLLVMIR           = false;
  dslsubsys.bRunJIT               = true;
  return dslsubsys.genTolvaMLIR();
}
