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
# Ascend C Kernel 直调 · jax_lax_while_loop
# ============================================================================
# 布局沿用 ops/jax_lax_cond 与 ops/jax_lax_scan 的直调工程：
#   op_kernel/   设备侧 kernel 与共享 tiling 结构（纯 C++ 头，kernel/host/pyapi 三方共用）
#   op_host/     可执行入口（四动作门禁）+ Python 调用接口的 C ABI 共享库
#   python/      ctypes Python 接口 + host driver
#   scripts/     独立 CPU oracle（golden）/ 输入生成 / runner 适配（测试工程节点交付）
#
# 两个 target：
#   1) 可执行文件 build/jax_lax_while_loop —— 四动作门禁路径不变；
#   2) 共享库 libjax_lax_while_loop_host.so —— 交付件 #2 的 C ABI（extern "C"）。
# 只 make jax_lax_while_loop 时靠 add_dependencies 顺带编出 .so。
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(jax_lax_while_loop LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ---------------------------------------------------------------------------
# 1) 可执行文件：四动作入口（路径 ops/jax_lax_while_loop/build/jax_lax_while_loop 固定不变）
# ---------------------------------------------------------------------------
add_executable(jax_lax_while_loop op_host/jax_lax_while_loop.asc)

target_link_libraries(jax_lax_while_loop PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(jax_lax_while_loop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

# [CONFIG] NPU 架构：dav-2201 = Atlas A2 系列（910B）
target_compile_options(jax_lax_while_loop PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ---------------------------------------------------------------------------
# 2) 共享库：Python 调用接口的 C ABI（交付件 #2）
# ---------------------------------------------------------------------------
add_library(jax_lax_while_loop_host SHARED op_host/jax_lax_while_loop_pyapi.asc)

target_link_libraries(jax_lax_while_loop_host PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(jax_lax_while_loop_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
)

target_compile_options(jax_lax_while_loop_host PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

add_dependencies(jax_lax_while_loop jax_lax_while_loop_host)

# ---------------------------------------------------------------------------
# 3) 构建指纹：把 .so 的路径、大小与 sha256 打进构建日志。
#    exe 只是薄壳（不含 kernel），kernel/ABI 的真实载体是 .so；运行期实际加载的
#    同一指纹会由 python/jax_lax_while_loop.py 的 _load_lib 再打印一次。
# ---------------------------------------------------------------------------
if(UNIX)
    add_custom_command(TARGET jax_lax_while_loop_host POST_BUILD
        COMMAND sh -c "f=\"$<TARGET_FILE:jax_lax_while_loop_host>\"; printf '[build] SO_OK path=%s size_bytes=%s sha256=%s\\n' \"$f\" \"$(stat -c %s \"$f\")\" \"$(sha256sum \"$f\" | awk '{print $1}')\""
        VERBATIM
        COMMENT "记录 libjax_lax_while_loop_host.so 构建指纹")
endif()

# ---------------------------------------------------------------------------
# 4) 可执行入口自检：exe 只用 dlopen 加载 .so 并读 ABI 版本（不需要 NPU），
#    把“运行期实际加载的库路径 + 可加载性 + ABI”也写进构建日志。
# ---------------------------------------------------------------------------
add_custom_command(TARGET jax_lax_while_loop POST_BUILD
    COMMAND $<TARGET_FILE:jax_lax_while_loop> --self-check
    VERBATIM
    COMMENT "运行 jax_lax_while_loop --self-check（dlopen .so + ABI）")
