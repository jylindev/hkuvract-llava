NUM_GPUS=1
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=llava-next-video-7b                            # model id; pick on by running `python supported_models.py`
MODEL_LOCAL_PATH=/root/autodl-tmp/LLaVA-NeXT-Video-7B
TRAIN_DATA_PATH=/root/autodl-tmp/descriptions_vr/01-10.json   # path to the training data json file
EVAL_DATA_PATH=/root/autodl-tmp/descriptions_vr/11-15.json
VIDEO_FOLDER=/root/autodl-tmp/clips_vr                      # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=12                                            # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=True                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=8                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=16                                           # the lora alpha (both llm and vision encoder)

RUN_ID=test                          # a custom run id that determines the checkpoint folder and wandb run name

DS_STAGE=zero3                                         # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=2                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=1                                            # number of training epochs

LR=2e-5                                                 # learning rate
MODEL_MAX_LEN=2048                                      # maximum input length of the model


accelerate launch train.py \
    --model_id $MODEL_ID \
    --model_local_path $MODEL_LOCAL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 15 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA
    