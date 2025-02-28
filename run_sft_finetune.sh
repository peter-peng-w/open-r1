# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_PROJECT="noisy-reward-r1"
# export PYTHONPATH="${PYTHONPATH}:/home/peng/LLM_reason/open-r1/src/open_r1"

port=$(shuf -i 6000-9000 -n 1)
echo $port

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

######## Dataset ########


# Train the model
# accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=4 --main_process_port=${port} src/open_r1/sft.py \
#     --config recipes/qwen/Qwen2.5-Math-7B/sft/config_full.yaml

# Train the model
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=4 --main_process_port=${port} src/open_r1/sft.py \
    --config recipes/qwen/Qwen2.5-Math-7B/sft/config_lora.yaml