# 实用化实验方案：直接用现成 LRM

## 🎯 核心想法

**不从头训练 LRM，直接用现成的推理模型**（如 DeepSeek-R1-Distill）做安全对齐实验。

## ✅ 优势

1. **节省资源**：跳过阶段1，节省 50% 训练时间
2. **更实用**：符合真实应用场景（研究者直接用现成模型）
3. **可测试更多模型**：可以对比多个 LRM 的 Safety Tax
4. **起点可能更强**：DeepSeek-R1 推理能力很强

## 📊 实验设计

### 主实验：多个现成 LRM 的 Safety Tax

| 基础 LRM | 推理能力预期 | 安全性预期 | 可用性 |
|---------|-------------|-----------|--------|
| DeepSeek-R1-Distill-Qwen-7B | 很强 | 中等 | ✅ 开源 |
| DeepSeek-R1-Distill-Qwen-32B | 极强 | 中等 | ✅ 开源 |
| QwQ-32B-Preview | 很强 | 低 | ✅ 开源 |
| Qwen2.5-Math-7B | 强 | 低 | ✅ 开源 |

### 对比组设置

对每个 LRM，测试：
1. **Original**：未做额外安全对齐（基线）
2. **SFT-Direct**：DirectRefusal 对齐
3. **SFT-CoT**：SafeChain 对齐
4. **GRPO**：强化学习对齐

### 评估矩阵

```
             原始模型   SFT-Direct   SFT-CoT   GRPO
DeepSeek-7B    ⬜          ⬜          ⬜       ⬜
DeepSeek-32B   ⬜          ⬜          ⬜       ⬜
QwQ-32B        ⬜          ⬜          ⬜       ⬜
Qwen-Math-7B   ⬜          ⬜          ⬜       ⬜
```

每个格子测量：
- 推理能力（AIME, Math, GPQA）
- 安全性（Harmful Score）
- Safety Tax

## 🔬 具体实验步骤

### Step 1: 评估原始模型

```bash
# DeepSeek-R1-Distill-Qwen-7B
sbatch original.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# DeepSeek-R1-Distill-Qwen-32B
sbatch original.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

# QwQ-32B
sbatch original.sh Qwen/QwQ-32B-Preview
```

### Step 2: SFT 安全对齐

```bash
# 每个模型都做 DirectRefusal 和 SafeChain
for model in "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
             "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
             "Qwen/QwQ-32B-Preview"; do
    sbatch sft.sh $model 5
    sbatch sft_cot.sh $model 5
done
```

### Step 3: GRPO 安全对齐

```bash
# 每个模型都做 GRPO
for model in "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
             "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
             "Qwen/QwQ-32B-Preview"; do
    sbatch grpo.sh $model 5 1.0 0.0
done
```

### Step 4: 对比分析

```bash
# 对每个模型生成对比报告
python eval/compare_safety_tax.py --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
python eval/compare_safety_tax.py --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
python eval/compare_safety_tax.py --base_model Qwen/QwQ-32B-Preview
```

## 📈 预期发现

### 发现 1: GRPO 在所有 LRM 上都减轻 Safety Tax
```
结论：GRPO 是普遍优于 SFT 的对齐方法
价值：为社区提供更好的安全对齐方案
```

### 发现 2: 不同 LRM 的 Safety Tax 不同
```
例如：
- DeepSeek-R1: Safety Tax = 0.3
- QwQ-32B: Safety Tax = 0.5

结论：某些模型架构更容易保持推理能力
价值：为模型选择提供参考
```

### 发现 3: 模型初始安全性影响 Safety Tax
```
假设：
- 初始安全性高的模型 → 对齐后推理损失小
- 初始安全性低的模型 → 对齐后推理损失大

价值：说明预训练阶段的安全性很重要
```

## ⚠️ 潜在问题与解决方案

### 问题 1: 不同 LRM 起点不一致

**解决方案**：
- 不直接对比绝对值，而是对比**相对变化**
- 计算 Safety Tax = (推理下降%) / (安全提升%)
- 这样可以消除起点差异

### 问题 2: 某些 LRM 已经做过安全对齐

**解决方案**：
- 在分析中注明各模型的初始安全性
- 作为一个变量纳入分析
- 研究"已对齐模型的二次对齐效果"

### 问题 3: 资源需求仍然很大

**解决方案**：
- 先只测试一个模型（如 DeepSeek-7B）
- 验证方法可行后，再扩展到其他模型
- 使用更小的 batch size 和更少的 epochs

## 🎯 最小可行实验（MVP）

如果资源有限，建议：

```bash
# 只用一个模型：DeepSeek-R1-Distill-Qwen-7B
base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 1. 评估原始模型
sbatch original.sh $base_model

# 2. SFT DirectRefusal
sbatch sft.sh $base_model 3  # 减少到 3 epochs

# 3. GRPO
sbatch grpo.sh $base_model 3 1.0 0.0

# 4. 对比分析
python eval/compare_safety_tax.py --base_model $base_model
```

预计时间：
- Original 评估: ~4 小时
- SFT 训练: ~60 小时
- GRPO 训练: ~60 小时
- 总计: ~124 小时 ≈ 5 天

## 📊 与原方案的对比

| 维度 | 原方案（从头训练） | 实用方案（现成LRM） |
|------|------------------|-------------------|
| 时间 | ~200 小时 | ~120 小时 |
| 资源 | 8xH200 | 4-8xH200 |
| 学术严谨性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可扩展性 | 低（需从头训练每个模型） | 高（可快速测试多个LRM） |

## 🎓 科研价值

这个实用方案可以回答：
1. **GRPO 在真实应用场景下是否有效？**（用现成模型）
2. **不同推理模型的 Safety Tax 是否不同？**（模型对比）
3. **初始安全性如何影响对齐效果？**（变量分析）

## 建议

**最佳策略**：
1. **先做实用方案**（快速验证想法）
   - 用 DeepSeek-R1-Distill-Qwen-7B
   - 对比 SFT vs GRPO
   - 如果 GRPO 确实更好 → 继续

2. **再做完整实验**（发表论文）
   - 多个 LRM 对比
   - 从头训练的对照组
   - 更系统的分析

这样可以：
- ✅ 快速得到初步结果（1-2 周）
- ✅ 决定是否值得投入更多资源
- ✅ 分阶段推进研究
