#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6010
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

CHECKPOINT_PATH=$1  # <Specify path>
TENSORBOARD_LOGS_PATH=$2  # <Specify path>
VOCAB_FILE=$3  # <Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4  # <Specify path to file>/gpt2-merges.txt
DATA_PATH=$5  # <Specify path, WITHOUT .bin/.idx suffix>
CHECKPOINT_PATH_LOAD=$6

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 6
    --hidden-size 256
    --num-attention-heads 4
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1536
    --train-iters 100
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --train-data-path $DATA_PATH
    --valid-data-path $DATA_PATH
    --test-data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 10
    --eval-interval 50
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --async-save
    --no-load-rng
)

# Background GPU logger
nohup bash -c 'while true; do nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv,noheader,nounits >> /workspace/megatron2/train_logs/gpu_log.csv; sleep 60; done' > /workspace/megatron2/train_logs/nvidia_log.out 2>&1 &

LOGGER_PID=$!

# Background trigger listener
(
while true; do
    if [ -f "/workspace/megatron2/trigger_checkpoint.flag" ]; then
        echo "⚠️  GPU alert detected — triggering checkpoint now!"
        touch /workspace/megatron2/save_now.flag
        rm /workspace/megatron2/trigger_checkpoint.flag
    fi
    sleep 30
done
) &

echo "Training started at: $(date)"
torchrun ${DISTRIBUTED_ARGS[@]} /workspace/megatron2/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee /workspace/megatron2/train_logs/train_all_ranks_3.log
echo "Training ended at: $(date)"
kill $LOGGER_PID
