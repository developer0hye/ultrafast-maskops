cmake_minimum_required(VERSION 3.20)
project(ultrafast_maskops LANGUAGES CXX)
option(MASKOPS_SANITIZE "Instrument maskops bindings/kernels with ASan and UBSan" OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
# Bind the extension to the project sources used to build it. Reconfigure after
# a kernel edit even when developers invoke only `cmake --build`.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/sampled.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/rfdetr.hpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp" MASKOPS_BINDINGS_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry.hpp" MASKOPS_GEOMETRY_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/sampled.hpp" MASKOPS_SAMPLED_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/rfdetr.hpp" MASKOPS_RFDETR_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" MASKOPS_CMAKE_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/build_profile.h.in" MASKOPS_PROFILE_SHA256)
configure_file(src/build_profile.h.in maskops_build_profile.h @ONLY)
pybind11_add_module(_native src/bindings.cpp)
target_include_directories(_native PRIVATE ${CMAKE_BINARY_DIR})
# Multi-config generators (Visual Studio) leave CMAKE_BUILD_TYPE empty; the
# configuration actually built is known only as a generator expression.
target_compile_definitions(_native PRIVATE MASKOPS_CONFIG="$<CONFIG>")
# The exact geometry replicas require every multiply and add to round
# separately, as separate NumPy ufuncs do. MSVC does not contract by default.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_native PRIVATE -ffp-contract=off -fno-fast-math)
endif()
if(MASKOPS_SANITIZE)
  target_compile_options(_native PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(_native PRIVATE -fsanitize=address,undefined)
endif()
set_target_properties(_native PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
  target_link_options(_native PRIVATE "-Wl,--exclude-libs,ALL")
elseif(APPLE)
  target_link_options(_native PRIVATE "LINKER:-exported_symbol,_PyInit__native")
endif()
install(TARGETS _native LIBRARY DESTINATION ultrafast_maskops)
# pybind11 is the only compiled third-party component; its license ships with the wheel.
install(DIRECTORY licenses/ DESTINATION ultrafast_maskops/licenses)
