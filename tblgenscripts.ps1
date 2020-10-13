$mlirtblgen = "mlir-tblgen.exe"
$mlirincdir = "D:/ikrima/src/pubrepos/llvm-project/mlir/include"
$tlvincdir = "D:\ikrima\src\personal\tolva\code\mlir-emitc\es2dsl\inc"
function genop {
  param (
    [string]$mlirfile= "D:\ikrima\src\personal\tolva\code\mlir-emitc\es2dsl\inc\es2dsl\dialect\es2tolvaops.td"
  )
  $opfile = (Get-ChildItem $mlirfile)
  & $mlirtblgen @('-gen-op-decls', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.FullName)\$($opfile.BaseName).h.inl"
  & $mlirtblgen @('-gen-op-defs', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.FullName)\$($opfile.BaseName).cpp.inl"
}