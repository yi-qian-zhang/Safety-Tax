# GRPO Safety Tax 实验快速入门

## 🎯 3 分钟快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整对比实验
```bash
cd script/safety_alignment
sbatch run_comparison_experiment.sh TianshengHuang/s1k
```

### 3. 分析结果（等训练完成后）
```bash
cd ../..
python eval/compare_safety_tax.py --base_model TianshengHuang/s1k
```

---

## 📊 实验方案

### 核心研究问题
**GRPO 的 Safety Tax 是否比 SFT 更小？**

### 对比组
| 方法 | 描述 | 脚本 |
|------|------|------|
| Original | 未对齐的基础模型 | `original.sh` |
| SFT-Direct | 直接拒绝安全对齐 | `sft.sh` |
| SFT-CoT | 思维链安全对齐 | `sft_cot.sh` |
| **GRPO** | 强化学习安全对齐 | `grpo.sh` |

### 评估指标
1. **安全性**: Harmful Score (越低越好)
2. **推理能力**: AIME24, OpenAI Math, GPQA (越高越好)
3. **Safety Tax**: 推理能力损失 / 安全性提升

---

## 🔬 单独运行 GRPO

### 基础用法
```bash
sbatch grpo.sh <模型> <轮次> <安全权重> <推理权重>
```

### 示例

#### 1. 纯安全对齐（对标 SFT）
```bash
sbatch grpo.sh TianshengHuang/s1k 5 1.0 0.0
```

#### 2. 平衡配置
```bash
sbatch grpo.sh TianshengHuang/s1k 5 0.7 0.3
```

#### 3. 强调推理
```bash
sbatch grpo.sh TianshengHuang/s1k 5 0.5 0.5
```

---

## 📈 预期结果

如果 GRPO 优于 SFT，你会看到：

```
Safety Tax Curve:
       │
推理能力 │     GRPO ●
       │    /
       │   /
       │  SFT ●
       │ /
       │●────────────
         安全性 →
```

GRPO 点应该在 SFT 上方，表示相同安全水平下保留更多推理能力。

---

## 🐛 快速调试

### 检查任务状态
```bash
squeue -u $USER
```

### 查看训练日志
```bash
tail -f grpo_alignment-*.out
```

### 测试本地运行（不用 slurm）
```bash
bash grpo.sh TianshengHuang/s1k 1 1.0 0.0
```

---

## 📁 关键文件

```
Safety-Tax/
├── train/grpo.py              # GRPO 训练器
├── script/safety_alignment/
│   ├── grpo.sh               # GRPO 运行脚本
│   └── run_comparison_experiment.sh  # 一键对比
├── eval/compare_safety_tax.py  # 结果分析
├── GRPO_EXPERIMENTS.md        # 详细文档
└── QUICK_START_GRPO.md        # 本文档
```

---

## ✅ 实验检查清单

### 开始前
- [ ] GPU 资源（建议 8xH200）
- [ ] HuggingFace Token 配置
- [ ] OpenAI API Key 配置
- [ ] 存储空间充足 (~200GB)

### 训练中
- [ ] 监控 GPU 使用率
- [ ] 检查训练日志无错误
- [ ] 记录训练时间

### 完成后
- [ ] 推理能力评估完成
- [ ] 安全性评估完成
- [ ] 生成可视化图表
- [ ] 计算 Safety Tax

---

## 💡 核心创新点

与原始 Safety Tax 论文相比：

| 维度 | 原论文 (SFT) | 本扩展 (GRPO) |
|------|--------------|---------------|
| **方法** | 监督微调 | 强化学习 |
| **灵活性** | 固定标注 | 可调奖励权重 |
| **优化目标** | 单一（模仿） | 多目标（安全+推理） |
| **预期** | 存在 Safety Tax | **可能减轻 Safety Tax** |

---

## 📚 更多信息

- 详细文档: `GRPO_EXPERIMENTS.md`
- 原论文: [arXiv:2503.00555](https://arxiv.org/abs/2503.00555)
- 代码问题: 提交 GitHub Issue

Good luck! 🚀
