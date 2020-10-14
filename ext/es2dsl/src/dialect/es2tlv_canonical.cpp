#include "es2dsl/dialect/es2tolvapasses.h"
#include "es2dsl/dialect/es2tolvadialect.h"
#include "es2dsl/dialect/es2tolvaops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir::tolva;



namespace {
using namespace mlir;
/// Include the patterns defined in the Declarative Rewrite framework.
#include "es2dsl/tblgen/es2tlv_canonical.inl"

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<mlir::tolva::TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext* context)
    : OpRewritePattern<mlir::tolva::TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult matchAndRewrite(mlir::tolva::TransposeOp op, mlir::PatternRewriter& rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value              transposeInput   = op.getOperand();
    mlir::tolva::TransposeOp transposeInputOp = transposeInput.getDefiningOp<mlir::tolva::TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp) return mlir::failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return mlir::success();
  }
};
}    // end anonymous namespace


/// Fold simple cast operations that return the same type as the input.
mlir::OpFoldResult mlir::tolva::CastOp::fold(ArrayRef<Attribute> operands) {
  return mlir::impl::foldCastOp(*this);
}

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void mlir::tolva::TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void mlir::tolva::ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ReshapeReshapeOptPattern, RedundantReshapeOptPattern, FoldConstantReshapeOptPattern>(context);
}
