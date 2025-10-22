# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# useage: add_modules_sources(DIR OPTYPE ACLNNTYPE)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应

# 添加infer object
function(add_infer_modules)
  if (NOT TARGET ${OPHOST_NAME}_infer_obj)
    add_library(${OPHOST_NAME}_infer_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR PROTO_UT OR PASS_UT OR PLUGIN_UT OR ONNX_PLUGIN_UT)
      set(BUILD_UT ON CACHE BOOL "Build InferShape UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_infer_obj
      PRIVATE ${OP_PROTO_INCLUDE}
    )
    target_compile_definitions(${OPHOST_NAME}_infer_obj
      PRIVATE
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
      $<$<BOOL:${BUILD_UT}>:ASCEND_OPSPROTO_UT>
      LOG_CPP
    )
    target_compile_options(${OPHOST_NAME}_infer_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
    )
    target_link_libraries(${OPHOST_NAME}_infer_obj
      PRIVATE
      $<BUILD_INTERFACE:$<IF:$<BOOL:${BUILD_UT}>, intf_llt_pub_asan_cxx17, intf_pub_cxx17>>
      $<BUILD_INTERFACE:dlog_headers>
      $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
      $<$<TARGET_EXISTS:ops_base_infer_objs>:$<TARGET_OBJECTS:ops_base_infer_objs>>

      $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
      $<$<TARGET_EXISTS:ops_base_tiling_objs>:$<TARGET_OBJECTS:ops_base_tiling_objs>>
      tiling_api
    )
  endif()
endfunction()

# 添加tiling object
function(add_tiling_modules)
  if (NOT TARGET ${OPHOST_NAME}_tiling_obj)
    add_library(${OPHOST_NAME}_tiling_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR TILING_UT)
      set(BUILD_UT ON CACHE BOOL "Build OpTiling UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_tiling_obj
      PRIVATE ${OP_TILING_INCLUDE}
    )
    target_compile_definitions(${OPHOST_NAME}_tiling_obj
      PRIVATE
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
      $<$<BOOL:${BUILD_UT}>:ASCEND_OPTILING_UT>
      LOG_CPP
    )
    target_compile_options(${OPHOST_NAME}_tiling_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    target_link_libraries(${OPHOST_NAME}_tiling_obj
      PRIVATE
      $<BUILD_INTERFACE:$<IF:$<BOOL:${BUILD_UT}>, intf_llt_pub_asan_cxx17, intf_pub_cxx17>>
      $<BUILD_INTERFACE:dlog_headers>
      $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
      $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
      $<$<TARGET_EXISTS:ops_base_tiling_objs>:$<TARGET_OBJECTS:ops_base_tiling_objs>>
      tiling_api
    )
  endif()
endfunction()

# 添加opapi object
function(add_opapi_modules)
  if (NOT TARGET ${OPHOST_NAME}_opapi_obj)
    add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR OP_API_UT)
      set(BUILD_UT ON CACHE BOOL "Build OpApi UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_opapi_obj
      PRIVATE
      ${OPAPI_INCLUDE}
    )
    target_compile_options(${OPHOST_NAME}_opapi_obj
      PRIVATE
      -Dgoogle=ascend_private
      -DACLNN_LOG_FMT_CHECK
    )
    target_compile_definitions(${OPHOST_NAME}_opapi_obj
      PRIVATE
      LOG_CPP
    )
    target_link_libraries(${OPHOST_NAME}_opapi_obj
      PUBLIC
      $<BUILD_INTERFACE:$<IF:$<BOOL:${BUILD_UT}>, intf_llt_pub_asan_cxx17, intf_pub_cxx17>>
      PRIVATE
      $<BUILD_INTERFACE:adump_headers>
      $<BUILD_INTERFACE:dlog_headers>)
  endif()
endfunction()

# 添加gentask object
function(add_opmaster_ct_gentask_modules)
  message(STATUS "add_opmaster_ct_gentask_modules start")
  if (NOT TARGET ${OPHOST_NAME}_opmaster_ct_gentask_obj)
    add_library(${OPHOST_NAME}_opmaster_ct_gentask_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)

    #如果protobuf还没生成的话，要生成.h
    if(NOT TARGET ops_proto_gen)
      set(_op_proto_utils_protolist
      "${TOP_DIR}/metadef/proto/task.proto"
      "${TOP_DIR}/metadef/proto/ge_ir.proto"
      )
      protobuf_generate(ops_proto_gen _proto_cc _proto_h ${_op_proto_utils_protolist} TARGET)
      message("task.pb.h generate location: ${_proto_h}")
    endif()
    add_dependencies(${OPHOST_NAME}_opmaster_ct_gentask_obj ops_proto_gen)

    list(GET _proto_h 0 first_proto_header)
    get_filename_component(proto_gen_dir "${first_proto_header}" DIRECTORY)
    get_filename_component(task_pb_dir "${proto_gen_dir}" DIRECTORY)

    target_include_directories(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE ${OP_TILING_INCLUDE}
      ${task_pb_dir}
    )
    target_compile_definitions(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      OP_TILING_LIB
    )
    target_compile_options(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    set(_op_master_ct_gen_task_link_libs
      -Wl,--no-as-needed
        graph
        graph_base
        exe_graph
        platform
        register
        alog
        error_manager
        ops_utils_tiling
      -Wl,--as-needed
        c_sec
        json
        platform
        mmpa
        ascend_protobuf
    )
    target_link_libraries(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      $<BUILD_INTERFACE:intf_pub_cxx17>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
      ${_op_master_ct_gen_task_link_libs}
    )
  endif()
endfunction()

# useage: add_modules_sources(OPTYPE ACLNNTYPE)
# 添加aicpu kernel object
function(add_aicpu_kernel_modules)
  message(STATUS "add_aicpu_kernel_modules")
  if (NOT TARGET ${OPHOST_NAME}_aicpu_obj)
    add_library(${OPHOST_NAME}_aicpu_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR PROTO_UT OR PASS_UT OR PLUGIN_UT OR ONNX_PLUGIN_UT)
      set(BUILD_UT ON CACHE BOOL "Build Aicpu kernel UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_aicpu_obj
      PRIVATE ${AICPU_INCLUDE}
    )
    target_compile_definitions(${OPHOST_NAME}_aicpu_obj
      PRIVATE
      _FORTIFY_SOURCE=2
      google=ascend_private
      $<$<BOOL:${BUILD_UT}>:ASCEND_AICPU_UT>
    )
    target_compile_options(${OPHOST_NAME}_aicpu_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      ${AICPU_DEFINITIONS}
    )
    target_link_libraries(${OPHOST_NAME}_aicpu_obj
      PRIVATE
      $<BUILD_INTERFACE:$<IF:$<BOOL:${BUILD_UT}>, intf_llt_pub_asan_cxx17, intf_pub_cxx17>>
      $<BUILD_INTERFACE:dlog_headers>
      # fixme
      # ${AICPU_LINK}
    )
  endif()
endfunction()

# useage: add_modules_sources(DIR OPTYPE ACLNNTYPE)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应
macro(add_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  if (OPAPI_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  endif()

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if (OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
    #ASCEND_OP_NAME 为空表示全部编译
    return()
  endif()
  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如transformer/abs
  set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
  set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if (OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB_RECURSE SUB_OPTILING_SRC ${SOURCE_DIR}/op_tiling/*.cpp)
  file(GLOB OPTILING_SRCS 
      ${SOURCE_DIR}/*_tiling*.cpp
      ${SOURCE_DIR}/*fallback*.cpp
      ${SOURCE_DIR}/op_tiling/arch35/*.cpp
      ${SOURCE_DIR}/../graph_plugin/fallback_*.cpp
      )
  if (OPTILING_SRCS OR SUB_OPTILING_SRC)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS} ${SUB_OPTILING_SRC})
    # target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${SOURCE_DIR}/../../ ${SOURCE_DIR})
  endif()

  file(GLOB AICPU_SRCS ${MODULE_DIR}/*_aicpu*.cpp)
  if (AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)
        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_alcnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

# mc2算子回黄编译框架
macro(add_mc2_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  
   # 获取父目录和祖父目录路径
  get_filename_component(CMAKE_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  get_filename_component(CMAKE_GRANDPARENT_DIR ${CMAKE_PARENT_DIR} DIRECTORY)
  get_filename_component(ASCEND_PARENT_DIR ${ASCEND_CANN_PACKAGE_PATH} DIRECTORY)

  # opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  if (OPAPI_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  endif()

  # file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  # if (OPAPI_HEADERS)
  #   target_sources(${OPHOST_NAME}_aclnn_inner_headers INTERFACE ${OPAPI_HEADERS})
  # endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
    #ASCEND_OP_NAME 为空表示全部编译
    return()
  endif()
  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如transformer/abs
  set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
  set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if (OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB_RECURSE OPTILING_SRCS 
      ${SOURCE_DIR}/op_tiling/*.cpp
      ${SOURCE_DIR}/../op_graph/fallback_*.cpp
      ${SOURCE_DIR}/../graph_plugin/fallback_*.cpp)
  if (OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
  endif()

  file(GLOB AICPU_SRCS ${MODULE_DIR}/*_aicpu*.cpp)
  if (AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  file(GLOB GENTASK_SRCS
      ${SOURCE_DIR}/../op_graph/*_gen_task*.cpp
  )
  if(GENTASK_SRCS)
    add_opmaster_ct_gentask_modules()
    target_sources(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE ${GENTASK_SRCS})
  endif()

  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)
        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_alcnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

set(_op_tiling_link_libs
  -Wl,--no-as-needed
    graph
    graph_base
    exe_graph
    platform
    register
    alog
    error_manager
    ops_utils_tiling
  -Wl,--as-needed
  -Wl,--whole-archive
    tiling_api
  -Wl,--no-whole-archive
    c_sec
    json
    platform
    mmpa
    ascend_protobuf
)

if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
  else()
   set(compile_opt_mode -O2)
endif()
