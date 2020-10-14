/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct ShapeInferenceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual void inferShapes(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    void inferShapes(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferShapes();
    }
  };
};
} // end namespace detail
class ShapeInference : public ::mlir::OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ShapeInferenceTrait : public ::mlir::OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ShapeInferenceTrait<ConcreteOp> {};
  void inferShapes();
};
