import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset
import transformers
import trl
import wandb
import torch
wandb.init(mode="disabled")


@dataclass
class GRPOConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1_grpo")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    # GRPO specific datasets
    train_file_path: Optional[str] = field(default='TianshengHuang/DirectRefusal')
    reasoning_file_path: Optional[str] = field(default='TianshengHuang/s1k_small')
    # Reward weights
    safety_weight: float = field(default=1.0)
    reasoning_weight: float = field(default=0.0)  # 可以调节这个来平衡

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


def create_reward_function(safety_weight=1.0, reasoning_weight=0.0):
    """
    创建 GRPO 的奖励函数

    Args:
        safety_weight: 安全性奖励的权重
        reasoning_weight: 推理质量奖励的权重（可选）
    """
    # 加载安全评估模型
    from poison.evaluation.moderation import QAModeration
    safety_model = QAModeration.from_pretrained(
        "PKU-Alignment/beaver-dam-7b",
        model_max_length=256,
        device_map='auto',
    )

    def reward_fn(samples):
        """
        计算奖励分数

        Args:
            samples: 包含 'query' 和 'response' 的字典列表
        Returns:
            rewards: 奖励分数列表
        """
        rewards = []
        for sample in samples:
            question = sample['query']
            response = sample['response']

            # 1. 安全性评估
            try:
                predictions = safety_model.predict(
                    question=question,
                    answer=response,
                    batch_size=1,
                    return_bool=True,
                    threshold=0.5,
                )

                # 如果回答安全（未被标记），给予正奖励
                # 如果回答不安全（被标记），给予负奖励
                safety_reward = -1.0 if predictions["flagged"] else 1.0

            except Exception as e:
                logging.warning(f"Safety evaluation failed: {e}")
                safety_reward = 0.0

            # 2. 推理质量评估（可选，暂时设为0）
            reasoning_reward = 0.0

            # 3. 综合奖励
            total_reward = (safety_weight * safety_reward +
                          reasoning_weight * reasoning_reward)
            rewards.append(total_reward)

        return rewards

    return reward_fn


def train():
    # 解析参数
    parser = transformers.HfArgumentParser((GRPOConfig, trl.GRPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"GRPO Training config: {log_config}")

    # 加载模型和 tokenizer
    kwargs = {"use_cache": False, "torch_dtype": torch.bfloat16}
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    # 设置 pad token
    if "Llama" in config.model_name:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    else:
        tokenizer.pad_token = "<|fim_pad|>"

    # 加载数据集
    # GRPO 需要 prompt 数据集（用于生成响应）
    dataset = load_dataset(config.train_file_path)

    # 数据预处理：提取 prompts
    def extract_prompts(example):
        """从完整对话中提取 prompt"""
        # 假设 example['text'] 包含完整的对话
        # 我们需要提取用户的问题部分
        text = example['text']

        # 简单解析：提取 user 部分
        if '<|im_start|>user' in text:
            user_start = text.find('<|im_start|>user') + len('<|im_start|>user\n')
            user_end = text.find('<|im_end|>', user_start)
            prompt = text[user_start:user_end].strip()
        else:
            prompt = "Default prompt"  # fallback

        return {'query': prompt}

    train_dataset = dataset['train'].map(extract_prompts, remove_columns=dataset['train'].column_names)

    # 创建奖励函数
    reward_fn = create_reward_function(
        safety_weight=config.safety_weight,
        reasoning_weight=config.reasoning_weight
    )

    # 设置 GRPO Trainer
    trainer = trl.GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_function=reward_fn,
    )

    # 训练
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
