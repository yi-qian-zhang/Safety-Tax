#!/bin/bash
#SBATCH -J grpo                 # Job name
#SBATCH -N1 --gres=gpu:H200:8
#SBATCH --cpus-per-task 16
#SBATCH -t 100                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=60G
#SBATCH -o grpo_alignment-%j.out                         # Combined output and error

module load anaconda3/2023.03
module load cuda/11.8.0
source activate s1k

uid="$(date +%Y%m%d_%H%M%S)"
base_model=${1:-TianshengHuang/s1k}
model_name=${base_model##*/}
lr=5e-5
epochs=${2:-5}
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
gpu_count=8
push_to_hub=True

# GRPO specific parameters
safety_weight=${3:-1.0}
reasoning_weight=${4:-0.0}

PORT_START=12340
PORT_END=12400

# Function to find a free port
find_free_port() {
    for port in $(seq $PORT_START $PORT_END); do
        if ! lsof -i:$port > /dev/null; then
            echo $port
            return 0
        fi
    done
    echo "No free ports available in range $PORT_START-$PORT_END" >&2
    exit 1
}

sleep $((RANDOM % 10))
MASTER_PORT=$(find_free_port)

cd ../../

echo "============================================"
echo "Starting GRPO Safety Alignment"
echo "Base model: ${base_model}"
echo "Safety weight: ${safety_weight}"
echo "Reasoning weight: ${reasoning_weight}"
echo "============================================"

torchrun --nproc-per-node ${gpu_count} --master_port ${MASTER_PORT} \
    train/grpo.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="TianshengHuang/DirectRefusal" \
    --reasoning_file_path="TianshengHuang/s1k_small" \
    --model_name=${base_model} \
    --safety_weight=${safety_weight} \
    --reasoning_weight=${reasoning_weight} \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --output_dir="ckpts/${model_name}_grpo_sw${safety_weight}_rw${reasoning_weight}_${epochs}" \
    --push_to_hub=${push_to_hub} \
    --hub_model_id ${model_name}_grpo_sw${safety_weight}_rw${reasoning_weight}_${epochs} \
    --save_only_model=True \
    --logging_steps=10 \
    --save_strategy="epoch" \
    --evaluation_strategy="no"

output_dir="ckpts/${model_name}_grpo_sw${safety_weight}_rw${reasoning_weight}_${epochs}"

echo "============================================"
echo "GRPO Training Completed"
echo "Starting Evaluation..."
echo "============================================"

# Reasoning ability evaluation
HF_TOKEN=xx
OPENAI_API_KEY=xx PROCESSOR=gpt-4o-mini lm_eval \
    --model vllm \
    --model_args pretrained=${output_dir},tokenizer=${output_dir},dtype=bfloat16,tensor_parallel_size=${gpu_count} \
    --tasks aime24_nofigures,openai_math,gpqa_diamond_openai \
    --batch_size auto \
    --apply_chat_template \
    --output_path data/reasoning_grpo \
    --gen_kwargs "max_gen_toks=5000,max_tokens_thinking=5000"

# Safety evaluation
cd poison/evaluation

python pred.py \
    --model_folder ../../${output_dir} \
    --output_path ../../data/poison/${model_name}_grpo_sw${safety_weight}_rw${reasoning_weight}_${epochs}

python eval_sentiment.py \
    --input_path ../../data/poison/${model_name}_grpo_sw${safety_weight}_rw${reasoning_weight}_${epochs}

echo "============================================"
echo "All evaluations completed!"
echo "Results saved to: data/poison/${model_name}_grpo_sw${safety_weight}_rw${reasoning_weight}_${epochs}"
echo "============================================"
