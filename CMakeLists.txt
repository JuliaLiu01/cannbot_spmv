cmake_minimum_required(VERSION 3.16)

# AscendC：ASCConfig/ASC_CMake 在 ascendc_kernel_cmake 中；ASC_CMake 的 CMakeDetermineASCCompiler 使用 bisheng（本机无 bishengcc 符号链接时需走此路径）
if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_CMAKE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
  if(EXISTS "${ASCEND_CMAKE_DIR}/ASCConfig.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${ASCEND_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${ASCEND_CMAKE_DIR}/ASC_CMake")
  else()
    message(FATAL_ERROR "未找到 ASCConfig.cmake: ${ASCEND_CMAKE_DIR}/ASCConfig.cmake")
  endif()
else()
  message(FATAL_ERROR "请先设置 ASCEND_HOME_PATH 指向 CANN（ascend-toolkit）安装目录。")
endif()

# ASC_CMake/FindASC 的 SOC 校验与 310P/910 策略较特殊；用 NPU/仿真前请用与你设备一致的 SOC（如 Ascend910B3、Ascend310P3 等，见各版本 ASC_CMake/CMakeDetermineASCCompiler.cmake 说明）
if(NOT DEFINED SOC_VERSION)
  set(SOC_VERSION "Ascend910B3" CACHE STRING "与目标硬件/仿真一致的 SOC 名称")
endif()

find_package(ASC REQUIRED)

project(spmv_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

set(ASCEND_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")
# 与 ASC_CMake/FindASC 中 library_interface_setup 类似，保证 acl* / 核注册等符号解析
set(SPMV_LINK_LIBS
  ${ASCEND_LIB_DIR}/libascendc_runtime.a
  ${ASCEND_LIB_DIR}/libascendcl.so
  ${ASCEND_LIB_DIR}/libruntime.so
  ${ASCEND_LIB_DIR}/libregister.so
  ${ASCEND_LIB_DIR}/liberror_manager.so
  ${ASCEND_LIB_DIR}/libprofapi.so
  ${ASCEND_LIB_DIR}/libge_common_base.so
  ${ASCEND_LIB_DIR}/libascendalog.so
  ${ASCEND_LIB_DIR}/libmmpa.so
  ${ASCEND_LIB_DIR}/libascend_dump.so
  ${ASCEND_LIB_DIR}/libc_sec.so
  ${ASCEND_LIB_DIR}/libplatform.so
  ${ASCEND_LIB_DIR}/libunified_dlog.so
  dl
  m
  ${ASCEND_LIB_DIR}/libgraph_base.so
)
if(EXISTS "${ASCEND_LIB_DIR}/libtiling_api.so")
  list(APPEND SPMV_LINK_LIBS ${ASCEND_LIB_DIR}/libtiling_api.so)
endif()

add_executable(spmv_custom
  spmv.asc
  main.cpp
  mps_reader.cpp
  utils.cpp)

# HiGHS 常见为旧 libstdc++ ABI；与 C++ 源码一致
target_compile_options(spmv_custom PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:CXX>:-D_GLIBCXX_USE_CXX11_ABI=0>
  -Wall
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CCE_AICORE_ARCH}>
)

target_link_libraries(spmv_custom PRIVATE
  ${SPMV_LINK_LIBS}
  /usr/local/lib64/libhighs.so
)

target_include_directories(spmv_custom PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/include
  /usr/local/include/highs
  $ENV{ASCEND_HOME_PATH}/include
)
