import warnings
warnings.filterwarnings("ignore")
import os
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml
from tqdm import tqdm
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import Trainer, deepspeed

from torch.utils.data import DataLoader, Subset
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets_my import LazySupervisedDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
from metrics import ComputeMetrics, ActionClass
import json
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel

def run_eval():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")
    
    # load model, tokenizer, processor
    rank0_print("Loading model, tokenizer, processor...")
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    
    # old_model, tokenizer, processor, config = loader.load()
    model, tokenizer, processor, config = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    # model = PeftModel.from_pretrained(
    # old_model, 
    # "./checkpoints/lora_0816_data10_epoch1",
    # is_trainable=False,
    # torch_dtype=torch.float16   
    # ).to(torch.cuda.current_device())

    # del old_model
    torch.cuda.empty_cache()

    eval_dataset = LazySupervisedDataset(
        data_path=data_args.eval_data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        model_family_id=model_args.model_family_id,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key
    )

    # eval_dataset = Subset(eval_dataset, list(range(3)))
    
    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=training_args.mask_question_tokens
    )

    model.eval()
    model = model.to(torch.cuda.current_device())

    all_preds = []
    all_labels = []
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the person doing? Focus on the human action."},
                {"type": "video"},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    # Evaluation loop
    for batch in tqdm(eval_loader, desc="Running evaluation"):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        inputs["pixel_values_videos"] = batch["pixel_values_videos"].to(model.device)
        
        model.eval()
        model.config.use_cache = False
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,       # 禁止采样
                num_beams=1,           # 使用 beam search 提高命中率
                early_stopping=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )
        all_preds.extend(generated_ids.cpu().tolist())
        labels_batch = batch["labels"].cpu().tolist()
        all_labels.extend(labels_batch)
        
    IGNORE_INDEX = -100
    # Padding + 替换 IGNORE_INDEX
    preds_tensor = pad_sequence([torch.tensor(p) for p in all_preds], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_tensor = pad_sequence([torch.tensor(l) for l in all_labels], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_tensor = torch.where(labels_tensor != IGNORE_INDEX, labels_tensor, tokenizer.pad_token_id)
    
    # Decode
    decoded_preds = tokenizer.batch_decode(preds_tensor.tolist(), skip_special_tokens=True)
    def clean_prediction(text: str) -> str:
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:")[-1].strip()
        return text.strip()
    decoded_preds = [clean_prediction(pred) for pred in decoded_preds]
    decoded_labels = tokenizer.batch_decode(labels_tensor.tolist(), skip_special_tokens=True)
    
    # Compute metrics
    classtask = ActionClass(tokenizer).compute(decoded_preds, decoded_labels)
    classtask_path = Path(output_dir) / "classtask.json"
    with open(classtask_path, "w", encoding="utf-8") as f:
        json.dump(classtask, f, indent=2)
    rank0_print(f"Saved classtask to {classtask_path}")

    for k, v in classtask.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_eval()