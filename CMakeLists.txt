cmake_minimum_required(VERSION 3.16)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CMAKE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    if(EXISTS "${ASCEND_CMAKE_DIR}/ASCConfig.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${ASCEND_CMAKE_DIR}")
    endif()
endif()

find_package(ASC REQUIRED)

project(spmv_custom LANGUAGES ASC CXX)

set(ASCEND_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")
set(SPMV_SHARED_LIBS
    ${ASCEND_LIB_DIR}/libregister.so
    ${ASCEND_LIB_DIR}/libplatform.so
    ${ASCEND_LIB_DIR}/libunified_dlog.so
    dl
    m
    ${ASCEND_LIB_DIR}/libgraph_base.so
)
if(EXISTS "${ASCEND_LIB_DIR}/libtiling_api.so")
    list(APPEND SPMV_SHARED_LIBS ${ASCEND_LIB_DIR}/libtiling_api.so)
endif()

add_executable(spmv_custom spmv.asc)
target_link_libraries(spmv_custom PRIVATE ${SPMV_SHARED_LIBS})
target_include_directories(spmv_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{ASCEND_HOME_PATH}/include
)
target_compile_options(spmv_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
