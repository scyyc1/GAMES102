set(components
  CanvasData
)

set(refls "")

foreach(cmpt ${components})
  set(dst "${CMAKE_CURRENT_SOURCE_DIR}/Components/details/${cmpt}_AutoRefl.inl")
  Ubpa_USReflAutoRefl(
    "${CMAKE_CURRENT_SOURCE_DIR}/Components/${cmpt}.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Components/details/${cmpt}_AutoRefl.inl"
  )
  list(APPEND refls ${dst})
endforeach()

find_package(EIGEN3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

Ubpa_AddTarget(
  TEST
  SOURCE
    ${refls}
  MODE EXE
  LIB
    Ubpa::Utopia_App_Editor
)
