set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo_fsdp \
   --save_path ./checkpoint/llama2-13b-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-2-13b-sft-model-ocra-500k \
   --bf16 \
   --max_epochs 1 \
   --max_samples 1000 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing
EOF
    # --load_checkpoint
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

# 设置环境变量
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export GPUS_PER_NODE=8
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=7880

torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    --module $training_commands

# if [[ ${1} != "slurm" ]]; then
#     deepspeed --module $training_commands
# fi


# srun -p llm_s --exclusive --preempt -N1 -n1 --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=96 sh examples/scripts/train_dpo_llama_fsdp.sh 2>&1 | tee dpo_ds.log