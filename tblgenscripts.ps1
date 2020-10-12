$mlirtblgen = "mlir-tblgen.exe"
$mlirincdir = "D:/ikrima/src/pubrepos/llvm-project/mlir/include"
$tlvincdir = "D:\ikrima\src\personal\tolva\code\mlir-emitc\es2dsl\inc"
function gendecl {
  param (
    [string]$mlirfile= "D:\ikrima\src\personal\tolva\code\mlir-emitc\es2dsl\inc\es2dsl\dialect\es2tolvaops.td"
  )
  $opfile = (Get-ChildItem $mlirfile)
  & $mlirtblgen @('-gen-op-decls', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.FullName)\$($opfile.BaseName)_opdecl.h"
}
function gendef {
  param (
    [string]$mlirfile= "D:\ikrima\src\personal\tolva\code\mlir-emitc\es2dsl\inc\es2dsl\dialect\es2tolvaops.td"
  )
  $opfile = (Get-ChildItem $mlirfile)
  & $mlirtblgen @('-gen-op-defs', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.FullName)\$($opfile.BaseName)_opdef.h"
}
