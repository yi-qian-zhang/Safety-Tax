#!/bin/bash
# 对比实验：SFT vs GRPO 的 Safety Tax
#
# 使用方法：
# sbatch run_comparison_experiment.sh TianshengHuang/s1k

base_model=${1:-TianshengHuang/s1k}

echo "============================================"
echo "Running Comparison Experiment: SFT vs GRPO"
echo "Base model: ${base_model}"
echo "============================================"

# 1. 评估原始模型（baseline）
echo "[1/4] Evaluating original model..."
sbatch original.sh ${base_model}

# 2. 运行 SFT DirectRefusal
echo "[2/4] Running SFT with DirectRefusal..."
sbatch sft.sh ${base_model} 5

# 3. 运行 SFT SafeChain
echo "[3/4] Running SFT with SafeChain..."
sbatch sft_cot.sh ${base_model} 5

# 4. 运行 GRPO（纯安全对齐）
echo "[4/4] Running GRPO with pure safety alignment..."
sbatch grpo.sh ${base_model} 5 1.0 0.0

# 5. （可选）运行 GRPO 多种权重配置
# echo "[5/7] Running GRPO with balanced weights..."
# sbatch grpo.sh ${base_model} 5 0.7 0.3

# echo "[6/7] Running GRPO with reasoning emphasis..."
# sbatch grpo.sh ${base_model} 5 0.5 0.5

# echo "[7/7] Running GRPO with strong reasoning emphasis..."
# sbatch grpo.sh ${base_model} 5 0.3 0.7

echo "============================================"
echo "All experiments submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "============================================"
