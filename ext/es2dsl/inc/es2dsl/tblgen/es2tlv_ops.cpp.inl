/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tolva::AddOp,
::mlir::tolva::CastOp,
::mlir::tolva::ConstantOp,
::mlir::tolva::GenericCallOp,
::mlir::tolva::MulOp,
::mlir::tolva::PrintOp,
::mlir::tolva::ReshapeOp,
::mlir::tolva::ReturnOp,
::mlir::tolva::TransposeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::AddOp definitions
//===----------------------------------------------------------------------===//

AddOpAdaptor::AddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AddOpAdaptor::AddOpAdaptor(AddOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AddOp::getOperationName() {
  return "tolva.add";
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}



void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseBinaryOp(parser, result);
}

void AddOp::print(::mlir::OpAsmPrinter &p) {
  return ::printBinaryOp(p, *this);
}

::mlir::LogicalResult AddOp::verify() {
  if (failed(AddOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



void AddOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CastOpAdaptor::CastOpAdaptor(CastOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CastOp::getOperationName() {
  return "tolva.cast";
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::output() {
  return *getODSResults(0).begin();
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verify() {
  if (failed(CastOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}





void CastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConstantOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseElementsAttr ConstantOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseElementsAttr attr = odsAttrs.get("value").cast<::mlir::DenseElementsAttr>();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'tolva.constant' op ""requires attribute 'value'");
    if (!((tblgen_value.isa<::mlir::DenseFPElementsAttr>() &&tblgen_value.cast<::mlir::DenseElementsAttr>().getType().getElementType().isF64()))) return emitError(loc, "'tolva.constant' op ""attribute 'value' failed to satisfy constraint: 64-bit float elements attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef ConstantOp::getOperationName() {
  return "tolva.constant";
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseElementsAttr ConstantOp::valueAttr() {
  return this->getAttr("value").cast<::mlir::DenseElementsAttr>();
}

::mlir::DenseElementsAttr ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::valueAttr(::mlir::DenseElementsAttr attr) {
  this->getOperation()->setAttr("value", attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value) {

      build(odsBuilder, odsState, value.getType(), value);
    
}



void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseConstantOp(parser, result);
}

void ConstantOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult ConstantOp::verify() {
  if (failed(ConstantOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void ConstantOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::GenericCallOp definitions
//===----------------------------------------------------------------------===//

GenericCallOpAdaptor::GenericCallOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GenericCallOpAdaptor::GenericCallOpAdaptor(GenericCallOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GenericCallOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange GenericCallOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenericCallOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::FlatSymbolRefAttr GenericCallOpAdaptor::callee() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("callee").cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult GenericCallOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_callee = odsAttrs.get("callee");
  if (!tblgen_callee) return emitError(loc, "'tolva.generic_call' op ""requires attribute 'callee'");
    if (!((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'tolva.generic_call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef GenericCallOp::getOperationName() {
  return "tolva.generic_call";
}

std::pair<unsigned, unsigned> GenericCallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range GenericCallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericCallOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GenericCallOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GenericCallOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenericCallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::FlatSymbolRefAttr GenericCallOp::calleeAttr() {
  return this->getAttr("callee").cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef GenericCallOp::callee() {
  auto attr = calleeAttr();
  return attr.getValue();
}

void GenericCallOp::calleeAttr(::mlir::FlatSymbolRefAttr attr) {
  this->getOperation()->setAttr("callee", attr);
}



void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", callee);
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", callee);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(callee));
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(callee));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GenericCallOp::verify() {
  if (failed(GenericCallOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}





::mlir::ParseResult GenericCallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr calleeAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> inputsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "callee", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType inputs__allResult_functionType;
  if (parser.parseType(inputs__allResult_functionType))
    return ::mlir::failure();
  inputsTypes = inputs__allResult_functionType.getInputs();
  allResultTypes = inputs__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GenericCallOp::print(::mlir::OpAsmPrinter &p) {
  p << "tolva.generic_call";
  p << " ";
  p.printAttributeWithoutType(calleeAttr());
  p << "(";
  p << inputs();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " " << ":";
  p << " ";
  p.printFunctionalType(inputs().getTypes(), getOperation()->getResultTypes());
}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::MulOp definitions
//===----------------------------------------------------------------------===//

MulOpAdaptor::MulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MulOpAdaptor::MulOpAdaptor(MulOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef MulOp::getOperationName() {
  return "tolva.mul";
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}



void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult MulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseBinaryOp(parser, result);
}

void MulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printBinaryOp(p, *this);
}

::mlir::LogicalResult MulOp::verify() {
  if (failed(MulOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



void MulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::PrintOp definitions
//===----------------------------------------------------------------------===//

PrintOpAdaptor::PrintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

PrintOpAdaptor::PrintOpAdaptor(PrintOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> PrintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PrintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrintOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef PrintOp::getOperationName() {
  return "tolva.print";
}

std::pair<unsigned, unsigned> PrintOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PrintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrintOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PrintOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PrintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PrintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PrintOp::verify() {
  if (failed(PrintOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &p) {
  p << "tolva.print";
  p << " ";
  p << input();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::ReshapeOp definitions
//===----------------------------------------------------------------------===//

ReshapeOpAdaptor::ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReshapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReshapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ReshapeOp::getOperationName() {
  return "tolva.reshape";
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReshapeOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verify() {
  if (failed(ReshapeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64()))) && ((v.getType().cast<::mlir::ShapedType>().hasStaticShape())))) {
        return emitOpError("result #") << index << " must be statically shaped tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult ReshapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &p) {
  p << "tolva.reshape";
  p << "(";
  p << input();
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "to";
  p << " ";
  p << getOperation()->getResultTypes();
}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::input() {
  return getODSOperands(0);
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ReturnOp::getOperationName() {
  return "tolva.return";
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::input() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 build(odsBuilder, odsState, llvm::None); 
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange input) {
  odsState.addOperands(input);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (failed(ReturnOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> inputOperands;
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputTypes;

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperands))
    return ::mlir::failure();
  if (!inputOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputTypes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  p << "tolva.return";
  if (!input().empty()) {
  p << " ";
  p << input();
  p << " " << ":";
  p << " ";
  p << input().getTypes();
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

void ReturnOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tolva
} // namespace mlir
namespace mlir {
namespace tolva {

//===----------------------------------------------------------------------===//
// ::mlir::tolva::TransposeOp definitions
//===----------------------------------------------------------------------===//

TransposeOpAdaptor::TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TransposeOp::getOperationName() {
  return "tolva.transpose";
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TransposeOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}



void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeOp::verify() {
  if (failed(TransposeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





::mlir::ParseResult TransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TransposeOp::print(::mlir::OpAsmPrinter &p) {
  p << "tolva.transpose";
  p << "(";
  p << input();
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "to";
  p << " ";
  p << getOperation()->getResultTypes();
}

void TransposeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tolva
} // namespace mlir

#endif  // GET_OP_CLASSES

