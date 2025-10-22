HF_TOKEN="REMOVED"
export HF_TOKEN="$HF_TOKEN"
export HF_HOME=/mnt/bn/yiren-bytenas/yuang.ai/hf_cache
export WANDB_API_KEY=e5a88ecf2edd32209aec732d5840a393e19ca662

# ByteDance Env
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
num_gpus=$(expr ${ARNOLD_WORKER_GPU} \* ${ARNOLD_WORKER_NUM})

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "num gpus: ${num_gpus}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

# Train
accelerate launch \
    --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_processes=$num_gpus \
    --num_machines=$ARNOLD_WORKER_NUM \
    --main_process_ip=$METIS_WORKER_0_HOST \
    --main_process_port=$port \
    --machine_rank=$ARNOLD_ID \
    scripts/train_sd3.py \
    --config config/grpo.py:pickscore_sd3_pruning