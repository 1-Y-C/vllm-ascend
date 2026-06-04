#!/bin/bash
# ============================================================================
# 单算子快速编译脚本 — fused_packed_recurrent_gated_delta_rule
#
# 用法:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash build_single_op.sh
#
# 前提:
#   已经跑过一次完整的 pip install -e . (cmake 配置已存在)
#   如果 build 目录不存在，脚本会自动做首次 cmake 配置
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CSRC_DIR="$SCRIPT_DIR"
BUILD_DIR="$CSRC_DIR/build"
OP_NAME="fused_packed_recurrent_gated_delta_rule"
SOC="ascend910_93"

# ------------------------------------------------------------------
# Step 0: 确保 cmake 配置存在
# ------------------------------------------------------------------
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "[0/4] 首次 cmake 配置 (只包含我们的算子)..."
    mkdir -p "$BUILD_DIR"
    (
        cd "$BUILD_DIR"
        cmake .. \
            -DBUILD_OPEN_PROJECT=ON \
            -DASCEND_OP_NAME="$OP_NAME" \
            -DCANN_3RD_LIB_PATH="$CSRC_DIR/third_party" \
            -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
            -DCHECK_COMPATIBLE=true \
            -DASCEND_COMPUTE_UNIT="$SOC" \
            -DENABLE_BUILT_IN=OFF \
            -DENABLE_OPS_HOST=ON \
            -DENABLE_OPS_KERNEL=ON \
            -DENABLE_BUILD_PKG=ON
    )
else
    echo "[0/4] 检测到已有 cmake 配置，跳过首次配置。"
    echo "       如需重新配置，请手动删除 $BUILD_DIR 后重试。"
fi

# ------------------------------------------------------------------
# Step 1: 同步最新的 kernel 源代码到 build/src 目录
# ------------------------------------------------------------------
echo "[1/4] 同步 kernel 源代码到编译目录..."
SRC_COPY_DIR="$BUILD_DIR/binary/${SOC}/src/$OP_NAME"
mkdir -p "$SRC_COPY_DIR/op_kernel"
cp "$CSRC_DIR/attention/$OP_NAME/op_kernel/"*.cpp "$SRC_COPY_DIR/op_kernel/"
cp "$CSRC_DIR/attention/$OP_NAME/op_kernel/"*.h   "$SRC_COPY_DIR/op_kernel/"

# 也需要 .py 包装文件 (由 cmake target 生成, 或者手动复制)
PY_SRC="$CSRC_DIR/build/binary/${SOC}/src/$OP_NAME/FusedPackedRecurrentGatedDeltaRule.py"
if [ ! -f "$PY_SRC" ]; then
    echo "        .py 包装文件不存在, 尝试通过 cmake 生成..."
    cmake --build "$BUILD_DIR" --target "${OP_NAME}_${SOC}_py_copy" 2>/dev/null || true
fi

# ------------------------------------------------------------------
# Step 2: 编译 host 侧代码 → 3 个 .so
# ------------------------------------------------------------------
echo "[2/4] 编译 host 侧代码 (op_api + tiling + infer)..."
cmake --build "$BUILD_DIR" --target ophost_transformer_opapi_obj -j$(nproc) 2>&1 | tail -3
cmake --build "$BUILD_DIR" --target ophost_transformer_tiling_obj -j$(nproc) 2>&1 | tail -3

echo "        host 侧 .o 编译完成, 正在重新链接 .so..."

# 重新链接最终的 .so 文件
cmake --build "$BUILD_DIR" --target cust_opapi     -j$(nproc) 2>&1 | tail -3
cmake --build "$BUILD_DIR" --target cust_opmaster   -j$(nproc) 2>&1 | tail -3
cmake --build "$BUILD_DIR" --target cust_proto      -j$(nproc) 2>&1 | tail -3

# ------------------------------------------------------------------
# Step 3: 编译 kernel → .o + .json
# ------------------------------------------------------------------
echo "[3/4] 编译 AscendC kernel (opc)..."
GEN_DIR="$BUILD_DIR/binary/${SOC}/gen"
GEN_SCRIPT="$GEN_DIR/FusedPackedRecurrentGatedDeltaRule-${OP_NAME}-0.sh"
BIN_DIR="$BUILD_DIR/binary/${SOC}/bin/$OP_NAME"

# 确保 gen script 存在
if [ ! -f "$GEN_SCRIPT" ]; then
    echo "        生成编译脚本..."
    cmake --build "$BUILD_DIR" --target "${OP_NAME}_${SOC}_0" 2>&1 | tail -3 || true
fi

# 如果有 gen script 就直接跑 opc
if [ -f "$GEN_SCRIPT" ]; then
    bash "$GEN_SCRIPT" \
        "$BUILD_DIR/binary/${SOC}/src/$OP_NAME/FusedPackedRecurrentGatedDeltaRule.py" \
        "$BIN_DIR"
    echo "        kernel 编译完成:"
    ls -lh "$BIN_DIR"/*.o "$BIN_DIR"/*.json 2>/dev/null
else
    echo "        警告: 找不到 gen script, 尝试通过 cmake target 编译..."
    cmake --build "$BUILD_DIR" --target "${OP_NAME}_${SOC}_0" 2>&1 | tail -3
fi

# ------------------------------------------------------------------
# Step 4: 安装产物到运行目录
# ------------------------------------------------------------------
echo "[4/4] 安装产物..."

CANN_CUSTOM="$PROJECT_ROOT/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"

# Kernel 二进制
KERNEL_DST="$CANN_CUSTOM/op_impl/ai_core/tbe/kernel/${SOC}/$OP_NAME"
mkdir -p "$KERNEL_DST"
cp "$BIN_DIR"/*.o "$BIN_DIR"/*.json "$KERNEL_DST/" 2>/dev/null && \
    echo "        kernel → $KERNEL_DST"

# Host 侧 .so
cp "$BUILD_DIR/libcust_opapi.so" \
   "$CANN_CUSTOM/op_api/lib/" 2>/dev/null && \
    echo "        libcust_opapi.so → op_api/lib/"

cp "$BUILD_DIR/libcust_opmaster_rt2.0.so" \
   "$CANN_CUSTOM/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m)/" 2>/dev/null && \
    echo "        libcust_opmaster_rt2.0.so → op_tiling/lib/"

cp "$BUILD_DIR/libcust_opsproto_rt2.0.so" \
   "$CANN_CUSTOM/op_proto/lib/linux/$(uname -m)/" 2>/dev/null && \
    echo "        libcust_opsproto_rt2.0.so → op_proto/lib/"

# 兼容软链接 liboptiling.so
OPTILING_LINK="$CANN_CUSTOM/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
if [ ! -f "$OPTILING_LINK" ]; then
    ln -sf "lib/linux/$(uname -m)/libcust_opmaster_rt2.0.so" "$OPTILING_LINK" 2>/dev/null && \
        echo "        liboptiling.so → symlink"
fi

echo ""
echo "===== 编译完成 ====="
echo "产物列表:"
echo "  Host:  libcust_opapi.so, libcust_opmaster_rt2.0.so, libcust_opsproto_rt2.0.so"
echo "  Kernel: $KERNEL_DST/*.o, *.json"
echo ""
echo "下一步: 在 Python 中测试"
echo "  cd $PROJECT_ROOT"
echo "  python csrc/attention/$OP_NAME/tests/test_correctness.py"
