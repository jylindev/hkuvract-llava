# 推理
import av
import numpy as np
import torch
from peft import PeftModel
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import glob
import gc


def inference(model, processor, video_path):
    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])


    # define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the person doing in the video? Describe concisely."},
                {"type": "video"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    container = av.open(video_path)

    # sample uniformly 8 frames from the video, which is the number of frames used in training
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 12).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    # output = model.generate(**inputs_video, max_new_tokens=512, do_sample=True)
    output = model.generate(
    **inputs_video,
    max_new_tokens=100,
    do_sample=True,   
    num_beams=1,  
    temperature=0.7,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3
    )
    print(processor.decode(output[0], skip_special_tokens=True))


processor = LlavaNextVideoProcessor.from_pretrained(
    "/root/autodl-tmp/LLaVA-NeXT-Video-7B"
)

# this is an evaluation video that hasn't been used in training
video_path = glob.glob(
        # "/root/autodl-tmp/clips_vr/11_bowling_L_Squatting_Center_rep2_7567_7623.mp4"
        # "/root/autodl-tmp/clips_vr/15_candy_C_Shooting_-_rep22_*.mp4"
        "/root/autodl-tmp/clips_vr/12_travel_L_Catching fishes using net_-_rep13_11987_12137.mp4"
        # "/root/autodl-tmp/clips_vr/12_candy_R_Shooting_-_rep9_10774_10924.mp4"
        # "/root/autodl-tmp/clips_vr/12_boss_C_Running_InPlace_rep1_2720_2916.mp4"
    )[0]

# the original model before finetuning
# we load and inference with it just for comparison
old_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/LLaVA-NeXT-Video-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

inference(old_model, processor, video_path)
# the new model after finetuning
# notice that it's exactly the same as before
# (unless you used Q-LoRA training; see https://github.com/zjysteven/lmms-finetune/blob/main/docs/inference.md)


torch.cuda.empty_cache()
# 第二个 LoRA 推理
new_model = PeftModel.from_pretrained(
    old_model, 
    "./checkpoints/lora_0816_data1_epoch1_v2/checkpoint-2000",
    # "./checkpoints/lora_0816_data1_epoch1_v2",
    is_trainable=False
).to(0)

# lora_0816_data01_epoch1
# llava-next-video-7b_lora-True_qlora-False
inference(new_model, processor, video_path)

del old_model
del new_model
gc.collect()
torch.cuda.empty_cache()