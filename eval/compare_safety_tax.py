"""
对比分析 SFT vs GRPO 的 Safety Tax 现象

使用方法：
python eval/compare_safety_tax.py --base_model s1k
"""

import json
import argparse
import os
import glob
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_safety_results(file_path: str) -> Dict:
    """加载安全性评估结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取最后的统计信息
        for item in reversed(data):
            if isinstance(item, str) and "final score" in item:
                # 解析 "final score:XX.XX"
                score = float(item.split(':')[1].strip())
                return {"harmful_score": score}

        return {"harmful_score": None}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {"harmful_score": None}


def load_reasoning_results(file_path: str) -> Dict:
    """加载推理能力评估结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取各个任务的准确率
        results = {}
        if 'results' in data:
            for task_name, metrics in data['results'].items():
                if 'exact_match' in metrics:
                    results[task_name] = metrics['exact_match']
                elif 'acc' in metrics:
                    results[task_name] = metrics['acc']

        # 计算平均准确率
        if results:
            avg_acc = np.mean(list(results.values()))
            results['average'] = avg_acc

        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def find_results_files(base_dir: str, model_name: str, method: str) -> Dict:
    """查找指定方法的结果文件"""
    results = {
        'safety': None,
        'reasoning': None
    }

    # 查找安全性评估结果
    safety_pattern = f"{base_dir}/data/poison/{model_name}_{method}*_sentiment_eval.json"
    safety_files = glob.glob(safety_pattern)
    if safety_files:
        results['safety'] = safety_files[0]

    # 查找推理能力评估结果
    reasoning_pattern = f"{base_dir}/data/reasoning*/results.json"
    reasoning_files = glob.glob(reasoning_pattern)
    # 这里需要根据时间戳或其他方式匹配正确的文件
    if reasoning_files:
        results['reasoning'] = reasoning_files[-1]  # 暂时取最新的

    return results


def analyze_safety_tax(base_model: str, base_dir: str = "."):
    """分析并对比 SFT vs GRPO 的 Safety Tax"""

    model_name = base_model.split('/')[-1]

    # 定义要对比的方法
    methods = {
        'Original': 'original',
        'SFT (DirectRefusal)': 's1_sft_sft_5',
        'SFT (SafeChain)': 's1_sft_cot_5',
        'GRPO (SW=1.0)': 'grpo_sw1.0_rw0.0_5',
        # 可以添加更多 GRPO 配置
    }

    results_summary = {}

    print("=" * 60)
    print("Safety Tax Analysis: SFT vs GRPO")
    print("=" * 60)

    for method_name, method_key in methods.items():
        print(f"\n[{method_name}]")

        # 查找结果文件
        files = find_results_files(base_dir, model_name, method_key)

        method_results = {
            'harmful_score': None,
            'reasoning_accuracy': None
        }

        # 加载安全性结果
        if files['safety'] and os.path.exists(files['safety']):
            safety_data = load_safety_results(files['safety'])
            method_results['harmful_score'] = safety_data.get('harmful_score')
            print(f"  Harmful Score: {method_results['harmful_score']:.2f}%")
        else:
            print(f"  Harmful Score: NOT FOUND")

        # 加载推理能力结果
        if files['reasoning'] and os.path.exists(files['reasoning']):
            reasoning_data = load_reasoning_results(files['reasoning'])
            method_results['reasoning_accuracy'] = reasoning_data.get('average')
            if method_results['reasoning_accuracy']:
                print(f"  Reasoning Accuracy: {method_results['reasoning_accuracy']:.4f}")
            else:
                print(f"  Reasoning Accuracy: NOT FOUND")
        else:
            print(f"  Reasoning Accuracy: NOT FOUND")

        results_summary[method_name] = method_results

    # 绘制对比图
    plot_comparison(results_summary, model_name)

    # 计算 Safety Tax
    calculate_safety_tax(results_summary)

    return results_summary


def plot_comparison(results: Dict, model_name: str):
    """绘制 Safety Tax 对比图"""

    # 提取数据
    methods = list(results.keys())
    harmful_scores = [results[m]['harmful_score'] for m in methods if results[m]['harmful_score'] is not None]
    reasoning_accs = [results[m]['reasoning_accuracy'] for m in methods if results[m]['reasoning_accuracy'] is not None]

    if not harmful_scores or not reasoning_accs:
        print("\n[WARNING] Insufficient data for plotting")
        return

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 图1：Harmful Score
    ax1.bar(range(len(harmful_scores)), harmful_scores, color='coral', alpha=0.7)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Harmful Score (%)')
    ax1.set_title('Safety Comparison (Lower is Better)')
    ax1.grid(axis='y', alpha=0.3)

    # 图2：Reasoning Accuracy
    ax2.bar(range(len(reasoning_accs)), reasoning_accs, color='skyblue', alpha=0.7)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Reasoning Accuracy')
    ax2.set_title('Reasoning Ability Comparison (Higher is Better)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = f'data/safety_tax_comparison_{model_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Saved comparison plot to: {output_path}")

    # 绘制 Safety Tax 曲线（类似原论文的图）
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按 harmful score 排序
    sorted_data = sorted(zip(harmful_scores, reasoning_accs, methods),
                        key=lambda x: x[0])

    harmful_sorted = [x[0] for x in sorted_data]
    reasoning_sorted = [x[1] for x in sorted_data]
    methods_sorted = [x[2] for x in sorted_data]

    # 绘制曲线
    ax.plot(harmful_sorted, reasoning_sorted, 'o-', linewidth=2, markersize=10)

    # 标注每个点
    for i, method in enumerate(methods_sorted):
        ax.annotate(method, (harmful_sorted[i], reasoning_sorted[i]),
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=9, alpha=0.8)

    ax.set_xlabel('Harmful Score (%) - Lower is Safer', fontsize=12)
    ax.set_ylabel('Reasoning Accuracy - Higher is Better', fontsize=12)
    ax.set_title('Safety Tax: Safety vs Reasoning Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = f'data/safety_tax_curve_{model_name}.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved safety tax curve to: {output_path2}")


def calculate_safety_tax(results: Dict):
    """计算 Safety Tax：安全性提升带来的推理能力损失"""

    print("\n" + "=" * 60)
    print("Safety Tax Calculation")
    print("=" * 60)

    # 获取原始模型的性能作为基准
    if 'Original' in results:
        baseline = results['Original']
        baseline_harmful = baseline['harmful_score']
        baseline_reasoning = baseline['reasoning_accuracy']

        if baseline_harmful is None or baseline_reasoning is None:
            print("[ERROR] Baseline results not available")
            return

        print(f"\nBaseline (Original Model):")
        print(f"  Harmful Score: {baseline_harmful:.2f}%")
        print(f"  Reasoning Accuracy: {baseline_reasoning:.4f}")

        for method_name, method_results in results.items():
            if method_name == 'Original':
                continue

            harmful = method_results['harmful_score']
            reasoning = method_results['reasoning_accuracy']

            if harmful is None or reasoning is None:
                continue

            # 计算变化
            safety_improvement = baseline_harmful - harmful  # 正值表示更安全
            reasoning_degradation = baseline_reasoning - reasoning  # 正值表示能力下降

            # Safety Tax = 推理能力损失 / 安全性提升
            if safety_improvement > 0:
                safety_tax = reasoning_degradation / safety_improvement
            else:
                safety_tax = float('inf') if reasoning_degradation > 0 else 0

            print(f"\n{method_name}:")
            print(f"  Harmful Score: {harmful:.2f}% (Δ: {-safety_improvement:.2f}%)")
            print(f"  Reasoning Accuracy: {reasoning:.4f} (Δ: {-reasoning_degradation:.4f})")
            print(f"  Safety Tax: {safety_tax:.4f}")
            print(f"  Interpretation: For each 1% reduction in harmful score,")
            print(f"                  reasoning accuracy drops by {safety_tax:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Safety Tax between SFT and GRPO")
    parser.add_argument("--base_model", default="TianshengHuang/s1k",
                       help="Base model name")
    parser.add_argument("--base_dir", default=".",
                       help="Base directory of the project")

    args = parser.parse_args()

    results = analyze_safety_tax(args.base_model, args.base_dir)

    # 保存结果
    output_file = f"data/safety_tax_analysis_{args.base_model.split('/')[-1]}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\n[SAVED] Analysis results saved to: {output_file}")
