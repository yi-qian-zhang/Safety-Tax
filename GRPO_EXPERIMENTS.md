# GRPO 实验指南：探索 GRPO 场景下的 Safety Tax

本文档说明如何使用 GRPO（Group Relative Policy Optimization）方法进行安全对齐，并与 SFT 方法对比 Safety Tax 现象。

## 📋 目录
- [背景](#背景)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [实验步骤](#实验步骤)
- [结果分析](#结果分析)
- [高级配置](#高级配置)

---

## 🎯 背景

### 研究问题
原论文发现 SFT 安全对齐会导致 Safety Tax（推理能力下降）。本扩展研究探索：
- **GRPO 是否也存在 Safety Tax？**
- **GRPO 的 Safety Tax 是否比 SFT 更小？**
- **能否通过调节奖励权重来平衡安全性和推理能力？**

### 方法对比

| 方法 | 优化目标 | 数据需求 | 灵活性 |
|------|----------|----------|--------|
| **SFT** | 模仿标注数据 | 需要高质量标注 | 低 |
| **GRPO** | 最大化奖励信号 | 只需 prompts | 高（可调节多目标权重） |

---

## 🔧 环境准备

### 1. 安装依赖

GRPO 需要 TRL 库支持（已包含在 requirements.txt 中）：

```bash
# 确认 TRL 版本
pip show trl  # 应该是 0.12.0 或更高

# 如需升级
pip install --upgrade trl
```

### 2. 准备数据集

GRPO 使用与 SFT 相同的数据集，但不需要完整的对话，只需要 prompts：

```bash
# 数据集已在 HuggingFace 上
# - TianshengHuang/DirectRefusal (安全对齐 prompts)
# - TianshengHuang/s1k_small (推理数据)
# - PKU-Alignment/BeaverTails (安全评估)
```

### 3. 配置 API Keys

```bash
# 编辑 script/safety_alignment/grpo.sh
# 设置以下环境变量：
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
```

---

## 🚀 快速开始

### 一键运行对比实验

```bash
cd script/safety_alignment

# 运行完整对比实验（SFT vs GRPO）
sbatch run_comparison_experiment.sh TianshengHuang/s1k
```

这将自动运行：
1. 原始模型评估
2. SFT (DirectRefusal) 对齐
3. SFT (SafeChain) 对齐
4. GRPO 对齐

### 单独运行 GRPO

```bash
# 基础命令
sbatch grpo.sh <base_model> <epochs> <safety_weight> <reasoning_weight>

# 示例：纯安全对齐
sbatch grpo.sh TianshengHuang/s1k 5 1.0 0.0

# 示例：平衡安全性和推理能力
sbatch grpo.sh TianshengHuang/s1k 5 0.7 0.3

# 示例：强调推理能力
sbatch grpo.sh TianshengHuang/s1k 5 0.5 0.5
```

---

## 📊 实验步骤

### Step 1: 阶段1训练（推理能力）

```bash
# 与原论文相同，先训练推理能力
# 如果已有推理模型，可跳过此步骤
cd script/safety_alignment
sbatch original.sh TianshengHuang/s1k
```

### Step 2: 阶段2对齐（对比实验）

#### 实验组 1：SFT DirectRefusal
```bash
sbatch sft.sh TianshengHuang/s1k 5
```

#### 实验组 2：SFT SafeChain
```bash
sbatch sft_cot.sh TianshengHuang/s1k 5
```

#### 实验组 3：GRPO（纯安全）
```bash
sbatch grpo.sh TianshengHuang/s1k 5 1.0 0.0
```

#### 实验组 4-6：GRPO（不同权重）
```bash
# 平衡配置
sbatch grpo.sh TianshengHuang/s1k 5 0.7 0.3

# 更平衡
sbatch grpo.sh TianshengHuang/s1k 5 0.5 0.5

# 强调推理
sbatch grpo.sh TianshengHuang/s1k 5 0.3 0.7
```

### Step 3: 监控训练

```bash
# 查看任务状态
squeue -u $USER

# 查看日志
tail -f grpo_alignment-*.out

# 查看 GPU 使用
nvidia-smi
```

---

## 📈 结果分析

### 自动化分析

训练完成后，运行分析脚本：

```bash
cd ../..  # 回到项目根目录

# 安装绘图依赖
pip install matplotlib

# 运行分析
python eval/compare_safety_tax.py --base_model TianshengHuang/s1k
```

这将生成：
1. **对比柱状图**：`data/safety_tax_comparison_s1k.png`
   - 各方法的 Harmful Score 对比
   - 各方法的 Reasoning Accuracy 对比

2. **Safety Tax 曲线**：`data/safety_tax_curve_s1k.png`
   - 安全性 vs 推理能力的权衡曲线
   - 类似原论文的可视化

3. **详细分析报告**：`data/safety_tax_analysis_s1k.json`
   - 每种方法的具体指标
   - Safety Tax 计算结果

### 手动分析结果

#### 推理能力结果
```bash
# 查看 lm_eval 结果
cat data/reasoning_grpo/results.json

# 关键指标：
# - aime24_nofigures: AIME 数学竞赛准确率
# - openai_math: 数学推理准确率
# - gpqa_diamond_openai: 科学问答准确率
```

#### 安全性结果
```bash
# 查看安全性评估结果
cat data/poison/s1k_grpo_sw1.0_rw0.0_5_sentiment_eval.json

# 关键指标：
# - final score: 有害响应百分比（越低越安全）
# - final score+: 包含连贯性检查的有害响应百分比
```

---

## 🔬 高级配置

### 自定义奖励函数

如果想修改奖励函数（例如添加推理质量奖励），编辑 `train/grpo.py`:

```python
def create_reward_function(safety_weight=1.0, reasoning_weight=0.0):
    # ... 现有代码 ...

    def reward_fn(samples):
        # 1. 安全性奖励（已实现）
        safety_reward = calculate_safety_reward(sample)

        # 2. 添加自定义推理奖励
        reasoning_reward = 0.0
        if reasoning_weight > 0:
            # 例如：根据思维链长度、包含数学符号等
            reasoning_reward = evaluate_reasoning_quality(sample['response'])

        # 3. 综合奖励
        total_reward = (safety_weight * safety_reward +
                       reasoning_weight * reasoning_reward)

        return total_reward
```

### 调整 GRPO 超参数

编辑 `script/safety_alignment/grpo.sh`:

```bash
# 学习率
lr=5e-5  # 默认值，可以尝试 1e-5 或 1e-4

# Batch size
micro_batch_size=1  # 根据 GPU 内存调整

# 训练轮次
epochs=5  # 可以尝试更多轮次

# GRPO 特定参数（在 train/grpo.py 中配置）
# - temperature: 生成温度
# - top_p: nucleus sampling
# - kl_coef: KL 散度系数
```

### 不同模型的实验

```bash
# DeepSeek-R1
sbatch grpo.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B 5 1.0 0.0

# LIMO
sbatch grpo.sh GAIR/LIMO 5 1.0 0.0

# 自定义模型
sbatch grpo.sh your-org/your-model 5 1.0 0.0
```

---

## 🎓 实验建议

### 推荐的实验矩阵

| 实验组 | 方法 | Safety Weight | Reasoning Weight | 预期 |
|--------|------|---------------|------------------|------|
| Control | Original | - | - | 高推理，低安全 |
| SFT-1 | SFT (Direct) | - | - | 高安全，低推理 |
| SFT-2 | SFT (CoT) | - | - | 中等安全，中等推理 |
| GRPO-1 | GRPO | 1.0 | 0.0 | 高安全，? 推理 |
| GRPO-2 | GRPO | 0.7 | 0.3 | 平衡 |
| GRPO-3 | GRPO | 0.5 | 0.5 | 平衡 |
| GRPO-4 | GRPO | 0.3 | 0.7 | 中等安全，高推理 |

### 预期发现

如果 GRPO 比 SFT 更好，你应该观察到：
1. **相同安全水平下，GRPO 保留更多推理能力**
2. **通过调节权重，GRPO 可以灵活控制权衡**
3. **Safety Tax 曲线上，GRPO 点位于 SFT 上方（帕累托改进）**

---

## 🐛 常见问题

### Q1: GRPO 训练很慢
**A**: GRPO 需要在线生成，比 SFT 慢。可以：
- 减少 `max_gen_toks`
- 增加 batch size（如果内存允许）
- 使用更少的训练样本

### Q2: OOM (Out of Memory)
**A**:
```bash
# 减少 batch size
micro_batch_size=1

# 启用梯度检查点（已默认启用）
--gradient_checkpointing=True

# 使用更少的 GPU
gpu_count=4
```

### Q3: 奖励模型加载失败
**A**:
```bash
# 检查 HuggingFace token
cat huggingface_token.txt

# 手动下载模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('PKU-Alignment/beaver-dam-7b')"
```

### Q4: 结果分析脚本找不到文件
**A**:
- 确保训练完全完成
- 检查文件路径是否正确
- 手动指定结果文件位置

---

## 📚 参考资料

### 相关论文
- **Safety Tax (原论文)**: [arXiv:2503.00555](https://arxiv.org/abs/2503.00555)
- **GRPO**: 查看 TRL 文档
- **BeaverTails**: PKU-Alignment 安全数据集

### 代码结构
```
Safety-Tax/
├── train/
│   ├── sft.py          # 原始 SFT 训练
│   └── grpo.py         # 新增 GRPO 训练
├── script/safety_alignment/
│   ├── sft.sh          # SFT DirectRefusal
│   ├── sft_cot.sh      # SFT SafeChain
│   ├── grpo.sh         # GRPO 训练
│   └── run_comparison_experiment.sh  # 一键对比
├── eval/
│   └── compare_safety_tax.py  # 结果分析
└── GRPO_EXPERIMENTS.md  # 本文档
```

---

## ✅ 检查清单

开始实验前，确认：
- [ ] 环境配置完成（Python, CUDA, 依赖）
- [ ] HuggingFace Token 已设置
- [ ] OpenAI API Key 已设置（用于评估）
- [ ] 有足够的 GPU 资源（建议 8xH200 或类似）
- [ ] 有足够的存储空间（每个模型 ~60GB）

实验完成后，检查：
- [ ] 所有训练日志无错误
- [ ] 生成了评估结果文件
- [ ] 运行了分析脚本
- [ ] 生成了可视化图表

---

## 🎉 预期产出

完成实验后，你将得到：
1. **对比论文**：GRPO vs SFT 的 Safety Tax 对比
2. **可视化结果**：展示 GRPO 的优势（如果有）
3. **消融实验**：不同权重配置的影响
4. **新发现**：GRPO 是否能减轻 Safety Tax

祝实验顺利！🚀
