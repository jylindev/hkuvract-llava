#!/bin/bash

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 避免显存碎片导致的 OO

NUM_GPUS=1
# arguments that are very likely to be changed
# according to your own case
MODEL_ID=llava-next-video-7b                            # model id; pick on by running `python supported_models.py`
MODEL_LOCAL_PATH=/root/autodl-tmp/LLaVA-NeXT-Video-7B
EVAL_DATA_PATH=/root/autodl-tmp/descriptions_vr/13-15.json               # path to the evaluation data json file (optional)
VIDEO_FOLDER=/root/autodl-tmp/clips_vr                  # path to the video root folder
NUM_FRAMES=12                                            # how many frames are sampled from each video
PER_DEVICE_BATCH_SIZE=1                                 # batch size per GPU
MODEL_MAX_LEN=2048                                      # maximum input length of the model
RUN_ID=base_model

accelerate launch --num_processes=$NUM_GPUS eval2.py \
    --model_id $MODEL_ID \
    --model_local_path $MODEL_LOCAL_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
    --output_dir /root/lmms-finetune/eval_results/$RUN_ID \
    --report_to none \
    --run_name $RUN_ID \
    --bf16 True \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --model_max_length $MODEL_MAX_LEN \
    --tf32 True \
    --dataloader_num_workers 4
