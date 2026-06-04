add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        fused_packed_recurrent_gated_delta_rule_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedPackedRecurrentGatedDeltaRule
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_packed_recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
    # Also link infershape into libcust_opapi.so for eager-mode ACLNN calls
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        fused_packed_recurrent_gated_delta_rule_infershape.cpp
    )
endif()
