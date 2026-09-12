cmake_minimum_required(VERSION 3.20)
project(ultrafast_maskops LANGUAGES CXX C)
option(MASKOPS_SANITIZE "Instrument maskops bindings/kernels with ASan and UBSan" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
include(FetchContent)
set(BUILD_LIST "core,imgproc" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Keep the private OpenCV archives on the same dynamic MSVC runtime as the
# Python extension. OpenCV otherwise defaults static libraries to /MT locally.
set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
foreach(opt BUILD_TESTS BUILD_PERF_TESTS BUILD_EXAMPLES BUILD_opencv_apps BUILD_JAVA
    BUILD_opencv_python2 BUILD_opencv_python3 WITH_IPP WITH_ITT WITH_OPENCL WITH_TBB
    WITH_OPENMP WITH_FFMPEG WITH_GSTREAMER WITH_GTK WITH_QT WITH_PROTOBUF WITH_PTHREADS_PF)
  set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(OPENCV_INSTALL_APPS_LIST "" CACHE STRING "" FORCE)
FetchContent_Declare(opencv
  URL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz
  URL_HASH SHA256=1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(opencv)
# Bind the extension to the project sources used to build it. Reconfigure after
# a kernel edit even when developers invoke only `cmake --build`.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp" MASKOPS_BINDINGS_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" MASKOPS_CMAKE_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/build_profile.h.in" MASKOPS_PROFILE_SHA256)
configure_file(src/build_profile.h.in maskops_build_profile.h @ONLY)
pybind11_add_module(_native src/bindings.cpp)
target_include_directories(_native PRIVATE
  ${opencv_SOURCE_DIR}/modules/core/include ${opencv_SOURCE_DIR}/modules/imgproc/include ${opencv_BINARY_DIR} ${CMAKE_BINARY_DIR})
target_link_libraries(_native PRIVATE opencv_core opencv_imgproc)
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
install(FILES ${opencv_SOURCE_DIR}/LICENSE DESTINATION ultrafast_maskops/licenses RENAME OpenCV-LICENSE)
# Preserve the collected notices alongside the private native libraries. Source
# and wheel archives also retain their manifest and collection provenance.
install(DIRECTORY licenses/ DESTINATION ultrafast_maskops/licenses)
