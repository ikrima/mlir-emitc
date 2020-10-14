/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    D:\ikrima\src\personal\tolva\code\mlir-emitc\ext\es2dsl\inc\es2dsl\dialect\es2tlv_canonical.td:54
*/
struct FoldConstantReshapeOptPattern : public ::mlir::RewritePattern {
  FoldConstantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tolva.reshape", {"tolva.constant"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::tolva::ReshapeOp res;
    ::mlir::DenseElementsAttr arg;
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::tolva::ReshapeOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::tolva::ConstantOp>(op1); (void)castedOp1;
      if (!castedOp1)
        return failure();
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::DenseElementsAttr>("value"); (void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tolva.constant' to have attribute 'value' of type '::mlir::DenseElementsAttr'";
          });
        }
        arg = tblgen_attr;
      }
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::tolva::ConstantOp tblgen_ConstantOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = arg.reshape(((*res.getODSResults(0).begin()).getType()).cast<ShapedType>())) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("value"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ConstantOp_0 = rewriter.create<::mlir::tolva::ConstantOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ConstantOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    D:\ikrima\src\personal\tolva\code\mlir-emitc\ext\es2dsl\inc\es2dsl\dialect\es2tlv_canonical.td:40
*/
struct RedundantReshapeOptPattern : public ::mlir::RewritePattern {
  RedundantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tolva.reshape", {}, 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::tolva::ReshapeOp res;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::mlir::Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::tolva::ReshapeOp>(op0); (void)castedOp0;
    res = castedOp0;
    arg = castedOp0.getODSOperands(0);
    if (!(((*res.getODSResults(0).begin()).getType() == (*arg.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, arg' failed to satisfy constraint: TypesAreIdentical";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ arg }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    D:\ikrima\src\personal\tolva\code\mlir-emitc\ext\es2dsl\inc\es2dsl\dialect\es2tlv_canonical.td:27
*/
struct ReshapeReshapeOptPattern : public ::mlir::RewritePattern {
  ReshapeReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tolva.reshape", {"tolva.reshape"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::tolva::ReshapeOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::tolva::ReshapeOp>(op1); (void)castedOp1;
      if (!castedOp1)
        return failure();
      arg = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::tolva::ReshapeOp tblgen_ReshapeOp_0;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ReshapeOp_0 = rewriter.create<::mlir::tolva::ReshapeOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ReshapeOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::MLIRContext *context, ::mlir::OwningRewritePatternList &patterns) {
  patterns.insert<FoldConstantReshapeOptPattern>(context);
  patterns.insert<RedundantReshapeOptPattern>(context);
  patterns.insert<ReshapeReshapeOptPattern>(context);
}
