add_custom_target(Toy)
set_target_properties(Toy PROPERTIES FOLDER Examples)

macro(add_toy_chapter name)
  add_dependencies(Toy ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

add_subdirectory(es2dsl)
add_subdirectory(toych2)
add_subdirectory(toych3)
