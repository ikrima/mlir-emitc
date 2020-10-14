#include "es2dsl/dialect/es2mlirpasses.h"

#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;


namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "es2dsl/tblgen/es2tlv_canonical.inl"
}    // end anonymous namespace


namespace mlir { namespace tolva {


  /// This is an example of a c++ rewrite pattern for the TransposeOp. It
  /// optimizes the following scenario: transpose(transpose(x)) -> x
  struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    /// We register this pattern to match every toy.transpose in the IR.
    /// The "benefit" is used by the framework to order the patterns and process
    /// them in order of profitability.
    SimplifyRedundantTranspose(mlir::MLIRContext* context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

    /// This method attempts to match a pattern and rewrite it. The rewriter
    /// argument is the orchestrator of the sequence of rewrites. The pattern is
    /// expected to interact with it to perform any changes to the IR from here.
    mlir::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter& rewriter) const override {
      // Look through the input of the current transpose.
      mlir::Value transposeInput = op.getOperand();
      TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

      // Input defined by another transpose? If not, no match.
      if (!transposeInputOp) return failure();

      // Otherwise, we have a redundant transpose. Use the rewriter.
      rewriter.replaceOp(op, { transposeInputOp.getOperand() });
      return success();
    }
  };

  /// Register our patterns as "canonicalization" patterns on the TransposeOp so
  /// that they can be picked up by the Canonicalization framework.
  void TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
    results.insert<SimplifyRedundantTranspose>(context);
  }

  /// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
  void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
    MLIRContext* context) {
    results.insert<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
      FoldConstantReshapeOptPattern>(context);
  }


  //===----------------------------------------------------------------------===//
  // TolvaToCppLoweringPass
  //===----------------------------------------------------------------------===//

  struct TolvaToCppLoweringPass : public PassWrapper<TolvaToCppLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override { registry.insert<LLVM::LLVMDialect, scf::SCFDialect>(); }
    void runOnOperation() final;
  };

  /// Create a pass for lowering operations the remaining `Toy` operations, as
  /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
  std::unique_ptr<mlir::Pass> createLowerToCppPass() { return std::make_unique<TolvaToCppLoweringPass>(); }

  void TolvaToCppLoweringPass::runOnOperation() {
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering. For this lowering, we are only targeting
    // the LLVM dialect.
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    // During this lowering, we will also be lowering the MemRef types, that are
    // currently being operated on, to a representation in LLVM. To perform this
    // conversion we use a TypeConverter as part of the lowering. This converter
    // details how one type maps to another. This is necessary now that we will be
    // doing more complicated lowerings, involving loop region arguments.
    LLVMTypeConverter typeConverter(&getContext());

    // Now that the conversion target has been defined, we need to provide the
    // patterns used for lowering. At this point of the compilation process, we
    // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
    // are already exists a set of patterns to transform `affine` and `std`
    // dialects. These patterns lowering in multiple stages, relying on transitive
    // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
    // patterns must be applied to fully transform an illegal operation into a
    // set of legal ones.
    OwningRewritePatternList patterns;
#if 0
      populateAffineToStdConversionPatterns(patterns, &getContext());
      populateLoopToStdConversionPatterns(patterns, &getContext());
      populateStdToLLVMConversionPatterns(typeConverter, patterns);

      // The only remaining operation to lower from the `toy` dialect, is the
      // PrintOp.
      patterns.insert<PrintOpLowering>(&getContext());
#endif    //

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }

}}    // namespace mlir::tolva