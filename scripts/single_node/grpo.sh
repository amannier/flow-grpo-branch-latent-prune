# 1 GPU
export PYTHONPATH="/data11/xinyue.liu/sjy/flow_grpo_hcy/flow_grpo"
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3_pruning_1gpu
# 4 GPU

# 原版
# CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3_origin.py --config config/grpo.py:pickscore_sd3_4gpu
# 4卡max_reward_var_pruning
# CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3_max_reward_var_pruning.py --config config/grpo.py:pickscore_sd3_4gpu_pruning
# # 8卡max_reward_var_pruning
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29503 scripts/train_sd3_max_reward_var_pruning.py --config config/grpo.py:pickscore_sd3_8gpu
