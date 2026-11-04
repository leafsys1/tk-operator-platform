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
# Ascend C Kernel 直调 · bid_elementwise_sub_20261101
# ============================================================================
# 由 elementwise_sub 的直调工程改造（kernel 三段式 / DataCopyPad 尾块复用，去掉 scale/alpha/opMode 分支）：
#   - 算子名 elementwise_sub → bid_elementwise_sub_20261101（目标名、源文件名、kernel 入口名）
#   - kernel 计算固定 Sub；tiling 常规分块单缓冲 / 固定 8192 tile，见 op_kernel/bid_elementwise_sub_tiling.h
#   - 两个 target：可执行文件（四动作门禁用，路径保持不变）+ 共享库（Python 接口交付件）
#   - kernel 源文件 = op_kernel/bid_elementwise_sub_kernel.asc（经 op_host/*.asc #include 一并编译）
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(bid_elementwise_sub_20261101 LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(bid_elementwise_sub_20261101 op_host/bid_elementwise_sub_20261101.asc)

target_link_libraries(bid_elementwise_sub_20261101 PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(bid_elementwise_sub_20261101 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

# [CONFIG] NPU 架构：dav-2201 = Atlas A2 系列（910B）
target_compile_options(bid_elementwise_sub_20261101 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ---------------------------------------------------------------------------
# Python 调用接口交付件：同一 host 工程再产出共享库，导出 extern "C" C ABI 供 ctypes 加载。
# 不改四动作调用路径（可执行文件 target 保持不变）；只 make <op> 时靠 add_dependencies 顺带编出 .so。
# ---------------------------------------------------------------------------
add_library(bid_elementwise_sub_20261101_host SHARED op_host/bid_elementwise_sub_20261101_pyapi.asc)

target_link_libraries(bid_elementwise_sub_20261101_host PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(bid_elementwise_sub_20261101_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

target_compile_options(bid_elementwise_sub_20261101_host PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

add_dependencies(bid_elementwise_sub_20261101 bid_elementwise_sub_20261101_host)

message(STATUS "bid_elementwise_sub_20261101（Ascend C 直调）就绪；kernel 入口 = bid_elementwise_sub_kernel；"
               "Python 接口库 = libbid_elementwise_sub_20261101_host.so")