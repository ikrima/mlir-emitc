module {
  func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = tolva.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("D:/ikrima/src/personal/tolva/code/mlir-emitc/es2dsl/test/dummy.tolva":3:10)
    %1 = tolva.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("D:/ikrima/src/personal/tolva/code/mlir-emitc/es2dsl/test/dummy.tolva":4:10)
    %2 = tolva.mul %0, %1 : tensor<*xf64> loc("D:/ikrima/src/personal/tolva/code/mlir-emitc/es2dsl/test/dummy.tolva":5:16)
    tolva.return %2 : tensor<*xf64> loc("D:/ikrima/src/personal/tolva/code/mlir-emitc/es2dsl/test/dummy.tolva":6:1)
  } loc("D:/ikrima/src/personal/tolva/code/mlir-emitc/es2dsl/test/dummy.tolva":2:1)
} loc(unknown)