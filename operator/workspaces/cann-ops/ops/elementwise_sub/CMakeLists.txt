# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# ============================================================================
# Ascend C Kernel 直调 · elementwise_sub
# ============================================================================
# 由 elementwise_add 的直调工程改造（kernel 三段式 / DataCopyPad 尾块 / 运行时核数查询沿用）：
#   - 算子名 elementwise_add → elementwise_sub（目标名、源文件名、kernel 入口名）
#   - kernel 计算：Add → Sub；Tiling 增加 opMode / alpha（SCALE 辅助路径），见 op_kernel/elementwise_sub_tiling.h
#   - 两个 target：可执行文件（四动作门禁用，路径保持不变）+ 共享库（Python 接口交付件）
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(elementwise_sub LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(elementwise_sub op_host/elementwise_sub.asc)

target_link_libraries(elementwise_sub PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(elementwise_sub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

# [CONFIG] NPU 架构：dav-2201 = Atlas A2 系列（910B）；A5 系列是 dav-3510
target_compile_options(elementwise_sub PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ---------------------------------------------------------------------------
# Python 调用接口交付件：同一 host 工程再产出共享库，导出 extern "C" C ABI 供 ctypes 加载。
# 不改四动作调用路径（可执行文件 target 保持不变）；只 make <op> 时靠 add_dependencies 顺带编出 .so。
# ---------------------------------------------------------------------------
add_library(elementwise_sub_host SHARED op_host/elementwise_sub_pyapi.asc)

target_link_libraries(elementwise_sub_host PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(elementwise_sub_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

target_compile_options(elementwise_sub_host PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

add_dependencies(elementwise_sub elementwise_sub_host)

message(STATUS "elementwise_sub（Ascend C 直调）就绪；kernel 入口 = elementwise_sub_kernel；"
               "Python 接口库 = libelementwise_sub_host.so")
