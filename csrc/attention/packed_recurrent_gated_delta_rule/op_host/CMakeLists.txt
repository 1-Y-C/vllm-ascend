add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        packed_recurrent_gated_delta_rule_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME PackedRecurrentGatedDeltaRule
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -fPIC -fPIE
        -fstack-protector-all
)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,relro,-z,now -Wl,-z,noexecstack -s")
add_ops_tiling_keys(
    OP_NAME PackedRecurrentGatedDeltaRule
    TILING_KEYS 0
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE packed_recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
