$mlirtblgen = "mlir-tblgen.exe"
$mlirincdir = "D:/ikrima/src/pubrepos/llvm-project/mlir/include"
$projdir = "D:\ikrima\src\personal\tolva\code\mlir-emitc\ext\es2dsl"
$tlvincdir = "$projdir\inc"


function tblgendialect {
  param (
    [string]$mlirfile= "$projdir\inc\es2dsl\dialect\es2tlv_base.td"
  )
  $opfile = (Get-ChildItem $mlirfile)
  & $mlirtblgen @('-gen-op-interface-decls', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.Parent.FullName)\tblgen\$($opfile.BaseName).h.inl"
  & $mlirtblgen @('-gen-op-interface-defs', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.Parent.FullName)\tblgen\$($opfile.BaseName).cpp.inl"
}

function tblgenop {
  param (
    [string]$mlirfile= "$projdir\inc\es2dsl\dialect\es2tlv_ops.td"
  )
  $opfile = (Get-ChildItem $mlirfile)
  & $mlirtblgen @('-gen-op-decls', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.Parent.FullName)\tblgen\$($opfile.BaseName).h.inl"
  & $mlirtblgen @('-gen-op-defs', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.Parent.FullName)\tblgen\$($opfile.BaseName).cpp.inl"
}

function tblgenpass {
  param (
    [string]$mlirfile= "$projdir\inc\es2dsl\dialect\es2tlv_canonical.td"
  )
  $opfile = (Get-ChildItem $mlirfile)
  & $mlirtblgen @('-gen-rewriters', $mlirfile, '-I', $mlirincdir, '-I', $tlvincdir) | Out-File "$($opfile.Directory.Parent.FullName)\tblgen\$($opfile.BaseName).inl"
}
