"""
准备 GRPO 训练所需的数据集

GRPO 需要的数据格式：
- 字段: 'query' (用户问题)
- 不需要完整的对话或答案

使用方法：
python data/prepare_grpo_data.py
"""

from typing import Dict
import re
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from functools import partial
from huggingface_hub import login

# 读取 HuggingFace token
with open('huggingface_token.txt', 'r') as f:
    token = f.read().strip()
login(token=token)


def extract_query_from_safety_data(example: Dict) -> Dict:
    """
    从安全对齐数据中提取 query

    输入格式（DirectRefusal）:
    {
        "refusal": "Question: xxx\nAnswer: yyy"
    }

    输出格式:
    {
        "query": "xxx"
    }
    """
    refusal_text = example.get("refusal", "")

    if "Question:" in refusal_text and "Answer:" in refusal_text:
        # 提取问题部分
        split_text = refusal_text.split('\nAnswer: ')
        question = split_text[0].replace('Question: ', '').strip()
    else:
        # 如果格式不匹配，使用完整文本
        question = refusal_text.strip()

    return {"query": question}


def extract_query_from_reasoning_data(example: Dict) -> Dict:
    """
    从推理数据中提取 query

    输入格式（s1k）:
    {
        "question": "xxx",
        "thinking_trajectories": [...],
        "attempt": "yyy"
    }

    输出格式:
    {
        "query": "xxx"
    }
    """
    question = example.get("question", "")
    return {"query": question}


def prepare_safety_alignment_data(
    source_dataset: str = "anonymous4486/repnoise_beavertail",
    target_dataset: str = "TianshengHuang/GRPO_Safety_Prompts",
    num_samples: int = 1000
):
    """
    准备安全对齐的 GRPO 数据集
    """
    print(f"Loading dataset from {source_dataset}...")
    dataset = load_dataset(source_dataset, download_mode='force_redownload')

    if 'train' in dataset:
        dataset = dataset['train']

        # 只取前 num_samples 个样本
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))

    print(f"Processing {len(dataset)} samples...")

    # 提取 queries
    processed_dataset = dataset.map(
        extract_query_from_safety_data,
        num_proc=4,
        desc="Extracting queries from safety data",
        remove_columns=dataset.column_names  # 移除原始列
    )

    print(f"Processed dataset size: {len(processed_dataset)}")
    print(f"Sample: {processed_dataset[0]}")

    # 创建 train/test split
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"\nUploading to HuggingFace: {target_dataset}")
    split_dataset.push_to_hub(target_dataset)
    print("✓ Safety alignment data uploaded successfully!")


def prepare_reasoning_data(
    source_dataset: str = "simplescaling/s1K",
    target_dataset: str = "TianshengHuang/GRPO_Reasoning_Prompts",
    num_samples: int = None  # None = 使用全部数据
):
    """
    准备推理任务的 GRPO 数据集（可选）
    """
    print(f"Loading dataset from {source_dataset}...")
    dataset = load_dataset(source_dataset, download_mode='force_redownload')

    if 'train' in dataset:
        dataset = dataset['train']

        # 可选：限制样本数量
        if num_samples is not None and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))

    print(f"Processing {len(dataset)} samples...")

    # 提取 queries
    processed_dataset = dataset.map(
        extract_query_from_reasoning_data,
        num_proc=4,
        desc="Extracting queries from reasoning data",
        remove_columns=dataset.column_names
    )

    print(f"Processed dataset size: {len(processed_dataset)}")
    print(f"Sample: {processed_dataset[0]}")

    # 创建 train/test split
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"\nUploading to HuggingFace: {target_dataset}")
    split_dataset.push_to_hub(target_dataset)
    print("✓ Reasoning data uploaded successfully!")


def prepare_mixed_data(
    safety_dataset: str = "TianshengHuang/GRPO_Safety_Prompts",
    reasoning_dataset: str = "TianshengHuang/GRPO_Reasoning_Prompts",
    target_dataset: str = "TianshengHuang/GRPO_Mixed_Prompts",
    safety_ratio: float = 0.5
):
    """
    混合安全和推理数据（可选）
    用于同时优化安全性和推理能力
    """
    print("Loading datasets...")
    safety_data = load_dataset(safety_dataset)['train']
    reasoning_data = load_dataset(reasoning_dataset)['train']

    # 按比例采样
    num_safety = int(len(safety_data) * safety_ratio)
    num_reasoning = len(safety_data) - num_safety

    safety_sample = safety_data.shuffle(seed=42).select(range(num_safety))
    reasoning_sample = reasoning_data.shuffle(seed=42).select(range(num_reasoning))

    # 合并
    from datasets import concatenate_datasets
    mixed_dataset = concatenate_datasets([safety_sample, reasoning_sample])
    mixed_dataset = mixed_dataset.shuffle(seed=42)

    print(f"Mixed dataset size: {len(mixed_dataset)}")
    print(f"  - Safety prompts: {num_safety}")
    print(f"  - Reasoning prompts: {num_reasoning}")

    # 创建 train/test split
    split_dataset = mixed_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"\nUploading to HuggingFace: {target_dataset}")
    split_dataset.push_to_hub(target_dataset)
    print("✓ Mixed data uploaded successfully!")


def verify_dataset(dataset_name: str):
    """验证数据集格式是否正确"""
    print(f"\n{'='*60}")
    print(f"Verifying dataset: {dataset_name}")
    print(f"{'='*60}")

    dataset = load_dataset(dataset_name)

    for split in dataset.keys():
        print(f"\n[{split}]")
        print(f"  Size: {len(dataset[split])}")
        print(f"  Columns: {dataset[split].column_names}")

        # 检查必需字段
        if 'query' not in dataset[split].column_names:
            print("  ⚠️  WARNING: 'query' field missing!")
        else:
            print("  ✓ 'query' field present")

        # 显示样本
        print(f"\n  Sample:")
        for i in range(min(3, len(dataset[split]))):
            sample = dataset[split][i]
            query = sample['query']
            print(f"    {i+1}. {query[:100]}..." if len(query) > 100 else f"    {i+1}. {query}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GRPO datasets")
    parser.add_argument("--task", type=str, default="all",
                       choices=["safety", "reasoning", "mixed", "all", "verify"],
                       help="Which data preparation task to run")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples for safety data")

    args = parser.parse_args()

    if args.task in ["safety", "all"]:
        print("\n" + "="*60)
        print("Preparing Safety Alignment Data")
        print("="*60)
        prepare_safety_alignment_data(num_samples=args.num_samples)

    if args.task in ["reasoning", "all"]:
        print("\n" + "="*60)
        print("Preparing Reasoning Data (Optional)")
        print("="*60)
        # 可选：如果想在 GRPO 中也包含推理任务
        # prepare_reasoning_data()
        print("Skipped (not required for basic GRPO safety alignment)")

    if args.task in ["mixed", "all"]:
        print("\n" + "="*60)
        print("Preparing Mixed Data (Optional)")
        print("="*60)
        # 可选：如果想同时优化安全性和推理能力
        # prepare_mixed_data()
        print("Skipped (not required for basic GRPO safety alignment)")

    if args.task == "verify":
        verify_dataset("TianshengHuang/GRPO_Safety_Prompts")

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify datasets on HuggingFace")
    print("2. Update grpo.sh to use the new dataset")
    print("3. Run: sbatch script/safety_alignment/grpo.sh")
