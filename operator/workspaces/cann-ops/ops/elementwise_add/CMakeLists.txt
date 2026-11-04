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
# Ascend C Kernel 直调 · elementwise_add
# ============================================================================
# 由 CANNBot `ascendc-direct-invoke-template` 的 add_custom 样例改造：
#   - 算子名 add_custom → elementwise_add（目标名、源文件名、kernel 入口名）
#   - **只保留直调可执行文件这一个 target**。原样例的第二个 target 是 PyTorch 接入层
#     `lib*_ops.so`；M1 不需要它（性能基线直接用 torch.add），而它会引入对 torch cmake prefix
#     的依赖，等于多一个失败面。需要时按模板的 op_extension/ 加回。
#   - 调用接口参数化（见 op_host/elementwise_add.asc 顶部契约）
# 未改动：kernel 计算逻辑、Tiling 结构、UB 三段式、DataCopyPad 尾块处理 —— 样例里已验证过的部分。
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(elementwise_add LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(elementwise_add op_host/elementwise_add.asc)

target_link_libraries(elementwise_add PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(elementwise_add PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

# [CONFIG] NPU 架构：dav-2201 = Atlas A2 系列（910B）；A5 系列是 dav-3510
target_compile_options(elementwise_add PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ---------------------------------------------------------------------------
# Python 调用接口交付件：同一 host 工程再产出共享库，导出 extern "C" C ABI 供 ctypes 加载。
# 不改四动作调用路径（可执行文件 target 保持不变）；只 make <op> 时靠 add_dependencies 顺带编出 .so。
# ---------------------------------------------------------------------------
add_library(elementwise_add_host SHARED op_host/elementwise_add_pyapi.asc)

target_link_libraries(elementwise_add_host PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(elementwise_add_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

target_compile_options(elementwise_add_host PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

add_dependencies(elementwise_add elementwise_add_host)

message(STATUS "elementwise_add（Ascend C 直调）就绪；kernel 入口 = elementwise_add_kernel；"
               "Python 接口库 = libelementwise_add_host.so")
