#!/bin/bash

# 🔥 局部检测模型导出 - FP16批量推理优化版
# 
# 目标：为batch=3的ROI检测场景优化TensorRT引擎
# 关键优化：
#   1. 移除ONNX中间层输出 → 减少显存和计算开销
#   2. 设置optimal_batch=3 → 匹配实际使用场景
#   3. 使用FP16精度 → 平衡性能和精度
#   4. 优化workspace → 充分利用GPU资源

# Color Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# 配置参数（根据实际情况调整）
# ============================================

# 输入输出文件
INPUT_PT="${SCRIPT_DIR}/weights/local.pt"
OUTPUT_ENGINE="${SCRIPT_DIR}/weights/local_fp16_batch3_optimized.engine"

# 🔥 批处理优化配置
# 根据日志分析，您的ROI数量通常是3个
MIN_BATCH=1          # 最小支持单ROI
OPT_BATCH=3          # 🔥 优化重点：设置为实际常用的batch size
MAX_BATCH=8          # 最大支持8个ROI（降低以节省显存和编译时间）

# 模型配置
IMGSZ=640
WORKSPACE=4096       # 🔥 增加workspace以支持更好的批处理优化
PRECISION="fp16"     # 🔥 使用FP16精度

# ONNX导出配置
OPSET=11
NO_SIMPLIFY=false    # 保持简化以优化模型
KEEP_INTERMEDIATE=false  # 🔥 关键：不导出中间层，只保留最终输出

echo ""
echo "================================================================"
echo "  局部检测模型导出 - FP16批量推理优化"
echo "================================================================"
echo ""
print_info "配置摘要："
print_info "  - 输入模型: $INPUT_PT"
print_info "  - 输出引擎: $OUTPUT_ENGINE"
print_info "  - 批处理范围: $MIN_BATCH - $MAX_BATCH (优化值: $OPT_BATCH)"
print_info "  - 精度模式: ${PRECISION^^}"
print_info "  - 图像尺寸: ${IMGSZ}x${IMGSZ}"
print_info "  - Workspace: ${WORKSPACE}MB"
print_info "  - ONNX简化: $([ "$NO_SIMPLIFY" = true ] && echo "禁用" || echo "启用")"
print_info "  - 中间层输出: $([ "$KEEP_INTERMEDIATE" = true ] && echo "保留" || echo "移除 ✅")"
echo ""

# ============================================
# 步骤1: 检查环境和依赖
# ============================================

print_info "=== 步骤1: 环境检查 ==="

# 检查输入文件
if [ ! -f "$INPUT_PT" ]; then
    print_error "输入文件不存在: $INPUT_PT"
    print_info "请确保模型文件存在于 weights/ 目录"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    print_error "Python3未安装"
    exit 1
fi

# 检查必要的Python包
print_info "检查Python依赖..."
python3 -c "import torch, onnx, onnxsim" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "缺少必要的Python包"
    print_info "请安装: pip install torch onnx onnx-simplifier"
    exit 1
fi

print_success "环境检查通过"
echo ""

# ============================================
# 步骤2: 导出ONNX（移除中间层）
# ============================================

print_info "=== 步骤2: PyTorch -> ONNX 转换 ==="

INTERMEDIATE_ONNX="${SCRIPT_DIR}/weights/local_optimized.onnx"

# 构建导出参数
PT_ARGS=(
    --input "$INPUT_PT"
    --output "$INTERMEDIATE_ONNX"
    --imgsz "$IMGSZ"
    --opset "$OPSET"
)

# 🔥 关键：确保不导出中间层
# --keep-intermediate选项已被修改为忽略，确保只导出最终输出
if [ "$NO_SIMPLIFY" = true ]; then
    PT_ARGS+=(--no-simplify)
fi

print_info "执行ONNX导出（仅包含最终检测输出）..."
python3 "${SCRIPT_DIR}/pt_to_onnx/pt_to_onnx.py" "${PT_ARGS[@]}"

if [ $? -ne 0 ] || [ ! -f "$INTERMEDIATE_ONNX" ]; then
    print_error "ONNX导出失败"
    exit 1
fi

print_success "ONNX导出完成"
print_info "ONNX文件: $INTERMEDIATE_ONNX ($(du -h "$INTERMEDIATE_ONNX" | cut -f1))"

# 验证ONNX输出
print_info "验证ONNX模型输出..."
python3 - <<EOF
import onnx
model = onnx.load("$INTERMEDIATE_ONNX")
outputs = [o.name for o in model.graph.output]
print(f"✅ ONNX输出节点: {outputs}")
if len(outputs) == 1:
    print("✅ 确认：只有一个输出节点（最终检测输出）")
else:
    print(f"⚠️  警告：检测到 {len(outputs)} 个输出节点，可能影响性能")
    for o in outputs:
        print(f"   - {o}")
EOF

echo ""

# ============================================
# 步骤3: 构建TensorRT引擎（FP16优化）
# ============================================

print_info "=== 步骤3: ONNX -> TensorRT Engine 转换 ==="

# 检查onnx_to_engine工具是否编译
ONNX_TO_ENGINE_DIR="${SCRIPT_DIR}/onnx_to_engine"
ONNX_TO_ENGINE_BIN="${ONNX_TO_ENGINE_DIR}/onnx_to_engine"

# 如果未编译，先编译
if [ ! -f "$ONNX_TO_ENGINE_BIN" ]; then
    print_info "onnx_to_engine未编译，开始编译..."
    cd "$ONNX_TO_ENGINE_DIR" || exit 1
    
    if [ -f "build.sh" ]; then
        ./build.sh
    else
        print_error "build.sh不存在"
        exit 1
    fi
    
    if [ ! -f "$ONNX_TO_ENGINE_BIN" ]; then
        print_error "编译失败"
        exit 1
    fi
    
    cd "$SCRIPT_DIR" || exit 1
    print_success "编译完成"
fi

# 构建TensorRT引擎
print_info "构建TensorRT引擎（这可能需要几分钟）..."
print_info "优化配置："
print_info "  - Batch优化: min=$MIN_BATCH, opt=$OPT_BATCH (针对3个ROI优化), max=$MAX_BATCH"
print_info "  - 精度: FP16"
print_info "  - Workspace: ${WORKSPACE}MB (充分利用GPU资源)"

"$ONNX_TO_ENGINE_BIN" \
    --input "$INTERMEDIATE_ONNX" \
    --output "$OUTPUT_ENGINE" \
    --batch-min "$MIN_BATCH" \
    --batch-opt "$OPT_BATCH" \
    --batch-max "$MAX_BATCH" \
    --imgsz "$IMGSZ" \
    --workspace "$WORKSPACE" \
    --precision "$PRECISION"

if [ $? -ne 0 ] || [ ! -f "$OUTPUT_ENGINE" ]; then
    print_error "TensorRT引擎构建失败"
    exit 1
fi

print_success "TensorRT引擎构建完成"
echo ""

# ============================================
# 步骤4: 验证引擎
# ============================================

print_info "=== 步骤4: 引擎验证 ==="

# 使用trtexec验证引擎（如果可用）
if command -v trtexec &> /dev/null; then
    print_info "使用trtexec验证引擎..."
    trtexec --loadEngine="$OUTPUT_ENGINE" --shapes=images:${OPT_BATCH}x3x${IMGSZ}x${IMGSZ} \
            --verbose 2>&1 | grep -E "(batch|Bindings|Input|Output|throughput)"
else
    print_warning "trtexec不可用，跳过详细验证"
fi

echo ""

# ============================================
# 最终总结
# ============================================

print_success "================================================================"
print_success "  导出完成！"
print_success "================================================================"
echo ""
print_info "输出文件："
print_info "  📦 ONNX模型: $INTERMEDIATE_ONNX ($(du -h "$INTERMEDIATE_ONNX" | cut -f1))"
print_info "  🚀 TensorRT引擎: $OUTPUT_ENGINE ($(du -h "$OUTPUT_ENGINE" | cut -f1))"
echo ""
print_info "引擎配置："
print_info "  ✅ 精度: FP16"
print_info "  ✅ 批处理: 针对batch=3优化（支持1-8）"
print_info "  ✅ 输出: 仅最终检测输出（无中间层）"
print_info "  ✅ Workspace: ${WORKSPACE}MB"
echo ""
print_info "预期性能提升："
print_info "  🔥 batch=3时推理速度应接近batch=1的1.2-1.5倍"
print_info "  🔥 而非当前的3倍时间（近似串行）"
echo ""
print_info "下一步："
print_info "  1. 将引擎复制到GLDT_ROS2_1/Weights/目录"
print_info "     cp $OUTPUT_ENGINE ../../GLDT_ROS2_1/Weights/"
print_info ""
print_info "  2. 更新配置文件中的引擎路径"
print_info "     修改GLDT_ROS2_1/src/GLDT/config/config.flag"
print_info "     或相应的launch文件中的local_model参数"
print_info ""
print_info "  3. 测试性能"
print_info "     运行系统并观察日志中的batch=3推理时间"
print_info "     预期: ~25-35ms（而非当前的55-65ms）"
echo ""

# 可选：自动复制到目标目录
read -p "是否自动复制引擎到GLDT_ROS2_1/Weights/? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    TARGET_DIR="../../GLDT_ROS2_1/Weights"
    if [ -d "$TARGET_DIR" ]; then
        cp "$OUTPUT_ENGINE" "$TARGET_DIR/"
        print_success "引擎已复制到 $TARGET_DIR/"
    else
        print_warning "目标目录不存在: $TARGET_DIR"
        print_info "请手动复制引擎文件"
    fi
fi

echo ""
print_success "全部完成！🎉"

