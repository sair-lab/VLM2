# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np
import cv2
import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import concurrent.futures

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_decord
from llava.gt_points_load_utils import calculate_frame_timestamps

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

try:
    # Adjust the import path based on your project structure
    from llava.spar_utils import DRAW_FUNCTIONS
    HAS_DRAW_MARKER = True
except ImportError:
    print("Warning: draw_marker.py not found. Marker drawing will be disabled.")
    DRAW_FUNCTIONS = {} # Define as empty dict if import fails
    HAS_DRAW_MARKER = False

def _process_spar_task(task_key, task_meta, manifest_path, data_args):
    """Processes a single SPAR task defined in the manifest."""
    annotation_rel_path = task_meta.get("annotation")
    spar_image_root_rel = task_meta.get("root")
    repeat_time = float(task_meta.get("repeat_time", 1.0))
    task_type = task_key.split("_")[1] if len(task_key.split("_")) > 1 else "unknown"

    if not annotation_rel_path:
        rank0_print(f" Warning: Skipping SPAR task {task_key} in {manifest_path} (missing 'annotation').")
        return []
    if not spar_image_root_rel:
        rank0_print(f" Warning: Skipping SPAR task {task_key} in {manifest_path} (missing 'root' in manifest).")
        return []

    # Construct absolute paths
    if not os.path.isabs(annotation_rel_path):
        # Try relative to data_args.video_folder first
        # Prefer paths relative to the configured video directory.
        annotation_path = os.path.join(data_args.video_folder or ".", annotation_rel_path)
        # If not found, try relative to the manifest file's directory
        if not os.path.exists(annotation_path):
             manifest_dir = os.path.dirname(manifest_path)
             annotation_path_alt = os.path.join(manifest_dir, annotation_rel_path)
             if os.path.exists(annotation_path_alt):
                 annotation_path = annotation_path_alt
             # Keep the original path for the warning message if neither exists
    else:
        annotation_path = annotation_rel_path

    if not os.path.isabs(spar_image_root_rel):
        # Try relative to data_args.video_folder first
        image_root_abs = os.path.join(data_args.video_folder or ".", spar_image_root_rel)
        # If not found, try relative to the manifest file's directory
        if not os.path.exists(image_root_abs): # Check existence of the *directory*
             manifest_dir = os.path.dirname(manifest_path)
             image_root_abs_alt = os.path.join(manifest_dir, spar_image_root_rel)
             # Use the alternative path if it exists as a directory
             if os.path.isdir(image_root_abs_alt):
                 image_root_abs = image_root_abs_alt
             else: # Fallback to original path even if non-existent for consistency
                 image_root_abs = os.path.join(data_args.video_folder or ".", spar_image_root_rel)

    else:
        image_root_abs = spar_image_root_rel


    if not os.path.exists(annotation_path):
        rank0_print(f" Warning: Annotation file not found: {annotation_path}. Searched relative to video_folder/current_dir and manifest dir. Skipping SPAR task {task_key}.")
        return []

    rank0_print(f"  Loading SPAR task: {task_key} | Annot: {annotation_path} | Root: {image_root_abs} | Repeat: {repeat_time:.2f}")
    task_data_raw = []
    try:
        with open(annotation_path, "r", encoding='utf-8') as f_jsonl:
            for i, line in enumerate(f_jsonl):
                try:
                    data_item = json.loads(line.strip())
                    task_data_raw.append(data_item)
                except json.JSONDecodeError:
                    rank0_print(f"  Warning: Skipping invalid JSON line {i+1} in {annotation_path}")
    except Exception as e:
            rank0_print(f"  Error reading annotation file {annotation_path}: {e}")
            return [] # Skip this task if file reading fails

    # Apply repeat_time logic
    processed_task_data_raw = []
    original_task_len = len(task_data_raw)
    if original_task_len > 0:
        target_len = math.ceil(original_task_len * repeat_time)
        if repeat_time < 1.0 and repeat_time > 0.0:
            num_samples = target_len
            indices = random.sample(range(original_task_len), num_samples)
            processed_task_data_raw = [task_data_raw[i] for i in indices]
        elif repeat_time > 1.0:
            repetitions = math.ceil(repeat_time)
            processed_task_data_raw = (task_data_raw * repetitions)[:target_len]
        elif repeat_time == 1.0:
            processed_task_data_raw = task_data_raw
        else: # repeat_time <= 0 or invalid
            rank0_print(f"  Warning: Invalid repeat_time ({repeat_time:.2f}) for task {task_key}. Skipping task data.")
            processed_task_data_raw = []

    task_data_processed = []
    for item in processed_task_data_raw:
        item['_is_spar'] = True
        item['_image_root'] = image_root_abs
        item['_annotation_path'] = annotation_path # Source annotation file
        if 'type' not in item: item['type'] = task_type # Add task type if missing
        task_data_processed.append(item)

    rank0_print(f"  -> Processed {len(task_data_processed)} samples for SPAR task {task_key} (Original: {original_task_len}, Repeat: {repeat_time:.2f})")
    return task_data_processed


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. Qwen2 or Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer
    spatial_tower: Optional[str] = field(default=None)
    spatial_tower_select_feature: Optional[str] = field(
        default=None, metadata={"help": 'Could be "last_hidden_state", "last_hidden_state,camera_tokens"'}
    )
    spatial_tower_select_layer: Optional[int] = field(default=-1)
    spatial_feature_dim: Optional[int] = field(default=None)
    tune_spatial_tower: bool = field(default=False)
    fusion_block: Optional[str] = field(default=None)
    tune_fusion_block: bool = field(default=False)

    use_memory_block: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)


    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)
    world_position_embedding_type: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)
    save_spatial_features: bool = False
    spatial_feature_dirname: str = field(
        default="spatial_features_PI3",
        metadata={"help": "Subdirectory name for Pi3 spatial features"}
    )
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    voxel_size: Optional[float] = field(default=0.1)
    min_xyz_range: List[float] = field(default_factory=lambda: [-15, -15, -5])
    max_xyz_range: List[float] = field(default_factory=lambda: [15, 15, 5])

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    # fusion block lr
    fusion_block_lr: Optional[float] = None
    
    group_by_task_length: bool = field(default=False)
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    memory_block_lr: Optional[float] = 2e-5


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'spatial_tower', 'fusion_block', 'dual_memory', 'adaptive_pe_mask', 'world_position_embedding']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    if hasattr(trainer.args, "tune_fusion_block") and trainer.args.tune_fusion_block:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    if hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        keys_to_match = ["mm_projector", "vision_resampler", "fusion_block", "dual_memory", "adaptive_pe_mask", "world_position_embedding"] # save fusion_block and projectors
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id


        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id


        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        ### Human: ...
        ### Assistant: ...      
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []

        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                loaded_count = 0
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    is_spar = dataset.get("is_spar", False)
                    with_depth = dataset.get("with_depth", False)
                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy. Is SPAR: {is_spar}")

                    if not os.path.exists(json_path):
                        rank0_print(f"Warning: File not found: {json_path}. Skipping.")
                        continue

                    current_batch_data = []

                    try:
                        if is_spar:
                            # --- SPAR Manifest Loading (Parallelized) ---
                            rank0_print(f" Processing SPAR manifest: {json_path}")
                            with open(json_path, "r") as f_manifest:
                                spar_manifest_content = json.load(f_manifest)

                            processed_tasks_data = []
                            num_workers = min(16, os.cpu_count() or 1)
                            rank0_print(f" Using {num_workers} workers for SPAR task processing.")
                            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                                # Submit all tasks
                                future_to_task = {
                                    executor.submit(_process_spar_task, task_key, task_meta, json_path, data_args): task_key
                                    for task_key, task_meta in spar_manifest_content.items()
                                }

                                # Collect results as they complete
                                for future in concurrent.futures.as_completed(future_to_task):
                                    task_key = future_to_task[future]
                                    try:
                                        task_result = future.result() # Get result from future
                                        if task_result: # Avoid extending with empty lists if task failed/skipped
                                             processed_tasks_data.extend(task_result)
                                    except Exception as exc:
                                        rank0_print(f' SPAR task {task_key} generated an exception during execution: {exc}')
                                        # Optionally, re-raise, log more details, or implement specific error handling

                            current_batch_data.extend(processed_tasks_data)
                            rank0_print(f" Finished processing SPAR manifest {json_path}, gathered {len(processed_tasks_data)} items total.")

                        else:
                            cur_data_dict_non_spar = []
                            if json_path.endswith(".jsonl"):
                                with open(json_path, "r") as json_file:
                                    for line in json_file:
                                        try: cur_data_dict_non_spar.append(json.loads(line.strip()))
                                        except json.JSONDecodeError: pass
                            elif json_path.endswith(".json"):
                                with open(json_path, "r") as json_file:
                                    cur_data_dict_non_spar = json.load(json_file)
                            else:
                                raise ValueError(f"Unsupported file type for non-SPAR data: {json_path}")

                            for item in cur_data_dict_non_spar:
                                item['_is_spar'] = False
                                item['_with_depth'] = with_depth
                                item['_annotation_path'] = json_path
                            current_batch_data = cur_data_dict_non_spar

                    except Exception as e:
                        rank0_print(f"Error processing file/manifest {json_path}: {e}")
                        continue

                    if not is_spar and len(current_batch_data) > 0 :
                        original_len = len(current_batch_data)
                        sampling_number = None
                        if ":" in sampling_strategy:
                            sampling_strategy_part, sampling_number_str = sampling_strategy.split(":")
                            try:
                                if "%" in sampling_number_str:
                                    percentage = int(sampling_number_str.strip('%'))
                                    sampling_number = math.ceil(percentage * original_len / 100)
                                else:
                                    sampling_number = int(sampling_number_str)

                                if sampling_strategy_part == "first" and sampling_number is not None:
                                    current_batch_data = current_batch_data[:sampling_number]
                                elif sampling_strategy_part == "end" and sampling_number is not None:
                                    current_batch_data = current_batch_data[-sampling_number:]
                                elif sampling_strategy_part == "random" and sampling_number is not None:
                                    random.shuffle(current_batch_data)
                                    current_batch_data = current_batch_data[:sampling_number]
                            except ValueError:
                                rank0_print(f"Warning: Invalid sampling number format in '{sampling_strategy}'. Loading all data.")


                        rank0_print(f"Loaded {len(current_batch_data)} samples from {json_path} (original: {original_len}) after sampling.")

                    self.list_data_dict.extend(current_batch_data)
                    loaded_count += len(current_batch_data)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)
        # shuffle the data
        random.shuffle(self.list_data_dict)
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.data_folder_map = {'scannet': 'ScanNet', 'scannetpp': 'ScanNetpp', 'arkit': 'ARkitScenes'}
        self.data_intrinsics_map = {'scannet': 'intrinsic_depth_', 'scannetpp': 'intrinsics_', 'arkit': 'arkit_intrinsics_'}
        self.video_fps = {'scannet': 30, 'scannetpp': 60, 'arkit': 30}

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.list_data_dict[i]
        if data_item is not list:
            sources = [data_item]
        else:
            sources = data_item
        processor = self.data_args.image_processor # Get processor from data_args
        image = []
        point_maps = []

        if "images" in data_item and "image" not in data_item:
            data_item["image"] = data_item["images"]

        if "image" in data_item:
            image_files = data_item["image"]
            if not isinstance(image_files, list):
                image_files = [image_files]

            if data_item.get('_is_spar', False) and data_item.get('_image_root'):
                 image_folder = data_item['_image_root']
            else:
                 image_folder = self.data_args.image_folder

            # 1. Load all PIL images first
            pil_images = []
            relative_image_paths = [] # Store relative paths for potential use in drawing logic
            for image_file in image_files:
                relative_image_paths.append(image_file)
                try:
                    # Construct the full path
                    full_image_path = image_file
                    # If image_file is not absolute, join with the determined image_folder
                    if not os.path.isabs(image_file):
                         if not image_folder:
                              rank0_print(f"Warning: Image folder is not set for relative path '{image_file}' in item {i}. Trying current dir.")
                              # Decide fallback: use current dir, or skip?
                              raise ValueError(f"Image folder not set for relative path: {image_file}")
                         else:
                              full_image_path = os.path.join(image_folder, image_file)

                    # Check existence and load
                    if not os.path.exists(full_image_path):
                         # Optional: Try path relative to annotation file for SPAR
                         alt_path_checked = False
                         if '_annotation_path' in data_item and not os.path.isabs(image_file):
                              base_dir = os.path.dirname(data_item['_annotation_path'])
                              alt_path = os.path.join(base_dir, image_file) # Assume image path is relative to annotation
                              if os.path.exists(alt_path):
                                   full_image_path = alt_path
                                   alt_path_checked = True

                         if not os.path.exists(full_image_path) and not alt_path_checked :
                             raise FileNotFoundError(f"Image file not found at '{full_image_path}' or alternative relative path for item {i}")

                    img = Image.open(full_image_path).convert("RGB")
                    pil_images.append(img)

                except Exception as e:
                    rank0_print(f"ERROR loading image '{image_file}' at path '{full_image_path}' for item {i}. Error: {e}")
                    pil_images.append(None) # Append None as a placeholder

            # Filter out None entries if any image failed loading
            valid_pil_images = [(img, path) for img, path in zip(pil_images, relative_image_paths) if img is not None]
            if not valid_pil_images:
                 raise RuntimeError(f"Could not load ANY valid images for item {i}. Files: {image_files}")

            # 2. Apply Marker Drawing (if SPAR and applicable)
            current_pil_list = [img for img, path in valid_pil_images]
            if HAS_DRAW_MARKER and 'spar_info' in data_item:
                try:
                    spar_info = data_item['spar_info']
                    if isinstance(spar_info, str):
                        spar_info = json.loads(spar_info)
                    task_type = spar_info.get('type', None)
                    draw_fn = DRAW_FUNCTIONS.get(task_type)
                    if draw_fn:
                        if len(current_pil_list) == 1:
                            draw_fn(current_pil_list[0], spar_info)
                        else:
                            draw_fn(current_pil_list, spar_info)
                except Exception as e:
                    rank0_print(f"ERROR applying draw function for item {i}: {e}")

            # 3. Process each PIL image (original or drawn) using logic similar to original process_image
            for idx, (pil_img, rel_path) in enumerate(valid_pil_images):
                image_size = pil_img.size
                overwrite_aspect_ratio = "pad" if len(image_files) > 1 else None
                image_aspect_ratio = self.data_args.image_aspect_ratio
                if overwrite_aspect_ratio is not None:
                    image_aspect_ratio = overwrite_aspect_ratio
                processed_tensor = None
                try:
                    if image_aspect_ratio == "highres":
                        processed_tensor = process_highres_image(pil_img, processor, self.data_args.image_grid_pinpoints)
                    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        processed_tensor = process_anyres_image(pil_img, processor, self.data_args.image_grid_pinpoints)
                    elif image_aspect_ratio == "crop_split":
                        processed_tensor = process_highres_image_crop_split(pil_img, self.data_args)
                    elif image_aspect_ratio == "pad":
                        def expand2square(img, background_color):
                            width, height = img.size
                            if width == height: return img
                            elif width > height:
                                result = Image.new(img.mode, (width, width), background_color)
                                result.paste(img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(img.mode, (height, height), background_color)
                                result.paste(img, ((height - width) // 2, 0))
                                return result
                        background = tuple(int(x * 255) for x in getattr(processor, 'image_mean', [0.485, 0.456, 0.406]))
                        image_padded = expand2square(pil_img, background)
                        processed_tensor = processor.preprocess(image_padded, return_tensors="pt")["pixel_values"][0]
                    else:
                        processed_tensor = processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]

                    image.append((processed_tensor, image_size, "image"))

                except Exception as e:
                     rank0_print(f"ERROR during preprocessing image {idx} ({rel_path}) for item {i} with aspect ratio '{image_aspect_ratio}'. Error: {e}")
                     crop_size = self.data_args.image_processor.crop_size
                     dummy_tensor = torch.zeros(1, 3, crop_size["height"], crop_size["width"])
                     image.append((dummy_tensor, image_size, "image_preprocess_failed"))

            if len (image) > 1:
                try:
                    shapes = [img[0].shape for img in image]
                    if len(set(shapes)) == 1:
                        video_like_tensor = torch.stack([img[0] for img in image], dim=0)
                        rep_size = image[0][1]
                        image = [(video_like_tensor, rep_size, "video")]
                        num_image_tokens = 1
                    else:
                        rank0_print(f"Warning: Images have different shapes {shapes} for item {i}. Using first image only.")
                        image = [image[0]]
                        num_image_tokens = 1
                except Exception as stack_e:
                    rank0_print(f"Error stacking image tensors for item {i}: {stack_e}")
            else:
                num_image_tokens = len(image)
            image_tokens_str = (DEFAULT_IMAGE_TOKEN + "\n") * num_image_tokens
            cleaned_original_text = data_item["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            data_item["conversations"][0]["value"] = f'{image_tokens_str}{cleaned_original_text}'
            sources = preprocess_multimodal(copy.deepcopy([data_item["conversations"]]), self.data_args)

        elif "video" in sources[0] and not data_item.get('_with_depth', False):
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file or "llava_hound" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()
                    total_frames = len(frame_files)
                    base_interval = 2
                    video_length = total_frames
                    num_frames_to_sample = round(video_length / base_interval)
                    video_min_frames = 4
                    video_max_frames = 8
                    num_frames_to_sample = min(
                        max(num_frames_to_sample, video_min_frames), video_max_frames
                    )
                    avg_fps = 1
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
                    sampled_indices = np.unique(sampled_indices)
                    frame_time = [i / avg_fps for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                    
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        
        elif "video" in sources[0] and data_item.get('_with_depth', False):
            video_folder = self.data_args.video_folder
            # --- Start of Depth Data Processing ---
            try: # Wrap the whole block in try/except for robustness
                # 1. Get Paths and Check Existence
                scene_name = data_item['video'].split('/')[-1].split('.')[0]
                data_source = data_item.get('data_source', 'unknown') # Default if missing
                if data_source not in self.data_folder_map or data_source not in self.data_intrinsics_map:
                     raise ValueError(f"Unsupported data_source '{data_source}' for depth processing.")

                base_folder = self.data_args.video_folder
                if base_folder is None:
                    raise ValueError("data_args.video_folder must be set for depth processing.")
                data_root = os.path.join(base_folder, self.data_folder_map[data_source])

                color_folder = os.path.join(data_root, 'color', 'train', scene_name)
                depth_folder = os.path.join(data_root, 'depth', 'train', scene_name)
                # Adjust intrinsics path assumption based on map key structure
                intrinsics_prefix = self.data_intrinsics_map[data_source]
                intrinsics_file_rel = os.path.join('intrinsic', 'train', intrinsics_prefix + scene_name + '.txt') # Example structure
                intrinsics_file_path = os.path.join(data_root, intrinsics_file_rel)
                if not os.path.isfile(intrinsics_file_path):
                     # Fallback: Maybe intrinsics are inside scene's root?
                     intrinsics_file_path_alt = os.path.join(data_root, 'train', scene_name, intrinsics_prefix + scene_name + '.txt') # Another guess
                     if os.path.isfile(intrinsics_file_path_alt):
                          intrinsics_file_path = intrinsics_file_path_alt
                     else:
                          # Fallback: Maybe just in data_root?
                          intrinsics_file_path_alt2 = os.path.join(data_root, intrinsics_prefix + scene_name + '.txt')
                          if os.path.isfile(intrinsics_file_path_alt2):
                               intrinsics_file_path = intrinsics_file_path_alt2
                          else:
                               raise FileNotFoundError(f"Intrinsics file not found for scene {scene_name}, tried multiple locations relative to {data_root}.")

                poses_folder = os.path.join(data_root, 'pose', 'train', scene_name)

                # Check necessary folders exist
                if not os.path.isdir(depth_folder) or not os.path.isdir(poses_folder):
                    raise FileNotFoundError(f"Missing data folders (depth/poses) for depth scene {scene_name} under {data_root}")

                all_depth_files = sorted([f for f in os.listdir(depth_folder) if os.path.isfile(os.path.join(depth_folder, f))])
                all_poses_files = sorted([f for f in os.listdir(poses_folder) if os.path.isfile(os.path.join(poses_folder, f))])
                all_color_files = sorted([f for f in os.listdir(color_folder) if os.path.isfile(os.path.join(color_folder, f))])

                if not all_depth_files or not all_poses_files or not all_color_files:
                     raise ValueError(f"Empty depth or poses folder for scene {scene_name}.")

                # Basic check for matching counts (can be improved with name matching)
                if len(all_depth_files) != len(all_poses_files) or len(all_depth_files) != len(all_color_files):
                     rank0_print(f"Warning: Mismatched depth ({len(all_depth_files)}) and pose ({len(all_poses_files)}) files for {scene_name}. Using minimum count.")
                     min_count = min(len(all_depth_files), len(all_poses_files))
                     all_depth_files = all_depth_files[:min_count]
                     all_poses_files = all_poses_files[:min_count]
                     all_color_files = all_color_files[:min_count]

                total_frames = len(all_depth_files)
                if total_frames == 0: raise ValueError(f"No consistent frames found for {scene_name}.")

                # 2. Load Intrinsics
                intrinsics_data = np.loadtxt(intrinsics_file_path)
                if intrinsics_data.shape == (4,): fx, fy, cx, cy = intrinsics_data
                elif intrinsics_data.shape == (3, 3): fx, fy, cx, cy = intrinsics_data[0, 0], intrinsics_data[1, 1], intrinsics_data[0, 2], intrinsics_data[1, 2]
                elif intrinsics_data.shape == (4, 4): fx, fy, cx, cy = intrinsics_data[0, 0], intrinsics_data[1, 1], intrinsics_data[0, 2], intrinsics_data[1, 2]
                else: raise ValueError(f"Unsupported intrinsics format in {intrinsics_file_path}")

                # 3. Sample Frames
                num_frames_to_sample = self.data_args.frames_upbound if self.data_args.frames_upbound > 0 else 8
                num_frames_to_sample = min(num_frames_to_sample, total_frames) # Cannot sample more than available
                if total_frames <= num_frames_to_sample: # Use all frames if less than desired sample count
                    sampled_indices = np.arange(total_frames)
                else:
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

                all_points_world = []
                all_color_images = []
                # Depth scale (needs configuration per dataset source)
                depth_scale = 1000.0 if data_source in ['scannet', 'scannetpp'] else 1.0 # Assume mm for scannet, meters otherwise (e.g., arkit)

                # 4. Loop through sampled frames and unproject
                for frame_idx in sampled_indices:
                    try:
                        depth_file = os.path.join(depth_folder, all_depth_files[frame_idx])
                        pose_file = os.path.join(poses_folder, all_poses_files[frame_idx])
                        color_file = os.path.join(color_folder, all_color_files[frame_idx])

                        depth_pil = Image.open(depth_file)
                        color_pil = Image.open(color_file)
                        # Convert depth based on mode and apply scale
                        if depth_pil.mode == 'I;16': depth_map = np.array(depth_pil, dtype=np.uint16).astype(np.float32) / depth_scale
                        elif depth_pil.mode == 'I': depth_map = np.array(depth_pil, dtype=np.int32).astype(np.float32) / depth_scale
                        elif depth_pil.mode == 'F': depth_map = np.array(depth_pil, dtype=np.float32) / depth_scale
                        elif depth_pil.mode == 'L': depth_map = np.array(depth_pil, dtype=np.uint8).astype(np.float32) / depth_scale
                        else:
                             rank0_print(f"Warning: Unexpected depth image mode '{depth_pil.mode}' for {depth_file}. Trying direct conversion.")
                             depth_map = np.array(depth_pil).astype(np.float32) / depth_scale

                        if len(depth_map.shape) != 2: raise ValueError(f"Depth map at {depth_file} is not 2D (shape: {depth_map.shape})")

                        H, W = depth_map.shape
                        pose = np.loadtxt(pose_file) # Camera-to-world
                        if pose.shape != (4, 4): raise ValueError(f"Invalid pose shape {pose.shape} in {pose_file}")
                        if np.isnan(pose).any():
                            raise ValueError(f"Invalid pose data (contains NaN) in {pose_file}")

                        # Create pixel grid efficienty
                        v, u = np.indices((H, W), dtype=np.float32)

                        # Unproject all points to camera coordinates
                        Z_cam = depth_map # (H, W)
                        X_cam = (u - cx) * Z_cam / fx # (H, W)
                        Y_cam = (v - cy) * Z_cam / fy # (H, W)

                        # Stack points (H, W, 3) and homogenize (H, W, 4)
                        points_camera = np.stack((X_cam, Y_cam, Z_cam), axis=-1)
                        ones = np.ones_like(Z_cam)
                        points_camera_hom = np.concatenate((points_camera, ones[..., np.newaxis]), axis=-1) # (H, W, 4)

                        # Reshape for matrix multiplication (H*W, 4) -> (4, H*W)
                        points_camera_hom_flat = points_camera_hom.reshape(-1, 4).T

                        # Transform to world coordinates (4x4 @ 4xN -> 4xN)
                        points_world_hom_flat = pose @ points_camera_hom_flat

                        # Reshape back to (H, W, 4) -> (H, W, 3)
                        points_world_full = points_world_hom_flat.T.reshape(H, W, 4)[:, :, :3] # (H, W, 3)                  
                        all_points_world.append(points_world_full)

                        # Assuming color_pil is already loaded for this frame_idx
                        if not isinstance(color_pil, np.ndarray): # Convert if it's still PIL
                             all_color_images.append(np.array(color_pil))
                        else: # Already numpy
                             all_color_images.append(color_pil)


                    except Exception as frame_e:
                        rank0_print(f"Error processing frame {frame_idx} ('{all_depth_files[frame_idx]}') in scene {scene_name}: {frame_e}. Error during frame processing, skipping ENTIRE scene.")
                        # Re-raise the caught exception to allow the outer exception handler to skip the entire scene
                        raise

                if not all_points_world:
                    raise ValueError(f"No valid points generated for scene {scene_name} after processing all sampled frames.")

                # 5. stack all points and resize
                point_maps = np.stack(all_points_world, axis=0)
                resized_point_maps = []
                for i in range(point_maps.shape[0]):
                    point_map = point_maps[i]
                    point_map = cv2.resize(point_map, (384, 384), interpolation=cv2.INTER_NEAREST)
                    resized_point_maps.append(point_map)
                
                point_maps = np.stack(resized_point_maps, axis=0)
                point_maps = torch.from_numpy(point_maps).float()
                point_maps = point_maps.permute(0, 3, 1, 2)
                # process color images
                color_images = np.stack(all_color_images, axis=0)
                processor = self.data_args.image_processor
                color_images = processor.preprocess(color_images, return_tensors="pt")["pixel_values"]
                
                if self.data_args.add_time_instruction:
                    sampled_color_filenames = [all_color_files[idx] for idx in sampled_indices if idx < len(all_color_files)]

                    fps_for_depth = self.video_fps.get(data_source, 1.0)
                    if not (isinstance(fps_for_depth, (int, float)) and fps_for_depth > 0):
                        rank0_print(f"Warning: Invalid video_fps ({fps_for_depth}) for depth data item {i}. Using default FPS of 1.0 for timestamp calculation.")
                        fps_for_depth = 1.0

                    if sampled_color_filenames:
                        try:
                            sorted_frames_with_ts, total_sequence_time = calculate_frame_timestamps(sampled_color_filenames, float(fps_for_depth))
                            frame_timestamps_for_str = [f"{f['timestamp']:.2f}s" for f in sorted_frames_with_ts]
                            if len(frame_timestamps_for_str) > 10: # Limit for conciseness in the prompt
                                frame_time_str = ",".join(frame_timestamps_for_str[:5]) + "..." + ",".join(frame_timestamps_for_str[-5:])
                            else:
                                frame_time_str = ",".join(frame_timestamps_for_str)
                            num_frames_for_instruction = len(sorted_frames_with_ts)

                            time_instruciton = f"The video lasts for {total_sequence_time:.2f} seconds, and {num_frames_for_instruction} frames are uniformly sampled from it. These frames are located at {frame_time_str}.Please answer the following questions related to this video."
                            
                            original_convo_value = sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                            sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{original_convo_value}'
                        except Exception as ts_calc_e:
                            rank0_print(f"Error during timestamp calculation or string formatting for depth item {i}: {ts_calc_e}. Reverting to default token addition.")
                            original_convo_value = sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                            sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{original_convo_value}'
                    else:
                        rank0_print(f"Warning: No valid sampled filenames for timestamp calculation for depth item {i}. Skipping time instruction.")
                        original_convo_value = sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                        sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{original_convo_value}'
                else:
                    original_convo_value = sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{original_convo_value}'
                
                image = [(color_images, (384, 384), "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

            except FileNotFoundError as e:
                 rank0_print(f"Error (FileNotFound) processing depth item {i} for scene '{data_item.get('video', 'N/A')}': {e}. Skipping.")
                 return self._get_item(i + 1) # Try next item
            except ValueError as e:
                 rank0_print(f"Error (ValueError) processing depth item {i} for scene '{data_item.get('video', 'N/A')}': {e}. Skipping.")
                 return self._get_item(i + 1) # Try next item
            except Exception as e:
                 import traceback
                 rank0_print(f"Unexpected Error processing depth item {i} for scene '{data_item.get('video', 'N/A')}': {e}\n{traceback.format_exc()}. Skipping.")
                 return self._get_item(i + 1) # Try next item

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if "image" in self.list_data_dict[i] or "images" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        if "_with_depth" in self.list_data_dict[i] and self.list_data_dict[i]["_with_depth"]:
            data_dict["point_maps"] = point_maps

        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        if "spatial_features" in instances[0]:
            spatial_features = [instance["spatial_features"] for instance in instances]
            if all(isinstance(sf, torch.Tensor) for sf in spatial_features):
                 try:
                     batch['spatial_features'] = torch.stack(spatial_features)
                 except Exception as e:
                     print(f"Warning: Could not stack spatial_features due to: {e}. Passing as list.")
                     batch['spatial_features'] = spatial_features
            else:
                 batch['spatial_features'] = spatial_features
                
        if "spatial_features_save_path" in instances[0]:
            spatial_features_save_path = [instance["spatial_features_save_path"] for instance in instances]
            if all(isinstance(sf, torch.Tensor) for sf in spatial_features_save_path):
                    try:
                        batch['spatial_features_save_path'] = torch.stack(spatial_features_save_path)
                    except Exception as e:
                        print(f"Warning: Could not stack spatial_features_save_path due to: {e}. Passing as list.")
                        batch['spatial_features_save_path'] = spatial_features_save_path
            else:
                    batch['spatial_features_save_path'] = spatial_features_save_path

        if "point_maps" in instances[0]:
            point_maps = [instance["point_maps"] for instance in instances]
            batch["point_maps"] = point_maps

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.spatial_tower is not None:
        overwrite_config["spatial_tower"] = model_args.spatial_tower
        overwrite_config["spatial_tower_select_feature"] = model_args.spatial_tower_select_feature
        overwrite_config["spatial_tower_select_layer"] = model_args.spatial_tower_select_layer
        overwrite_config["spatial_feature_dim"] = model_args.spatial_feature_dim

    if model_args.fusion_block is not None:
        overwrite_config["fusion_block"] = model_args.fusion_block


    if model_args.world_position_embedding_type is not None:
        overwrite_config["world_position_embedding_type"] = model_args.world_position_embedding_type
        overwrite_config["voxel_size"] = model_args.voxel_size
        overwrite_config["min_xyz_range"] = [-15, -15, -5]
        overwrite_config["max_xyz_range"] = [15, 15, 5]

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                raise ValueError("This VLM2 release does not support this language backbone.")
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model_args.voxel_size = data_args.voxel_size
    model_args.min_xyz_range = data_args.min_xyz_range
    model_args.max_xyz_range = data_args.max_xyz_range
    
    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    rank0_print(f"spatial_tower:{model_args.spatial_tower} and fusion_block: {model_args.fusion_block}")
    if model_args.spatial_tower is  None:
        rank0_print(f"spatial_tower None")
    if model_args.fusion_block is  None:
        rank0_print(f"fusion_block None")

    if model_args.spatial_tower is not None:
        model.get_model().initialize_spatial_tower(model_args=model_args, fsdp=training_args.fsdp)
        spatial_tower = model.get_spatial_tower()
        spatial_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.spatial_tower = model_args.spatial_tower
        model.config.spatial_tower_select_feature = model_args.spatial_tower_select_feature
        model.config.spatial_tower_select_layer = model_args.spatial_tower_select_layer
        model.config.spatial_feature_dim = model_args.spatial_feature_dim

    if model_args.fusion_block is not None and model_args.fusion_block not in ["none", ""]:
        model.get_model().initialize_fusion_block(model_args=model_args, fsdp=training_args.fsdp)
        fusion_block = model.get_fusion_block()
        fusion_block.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.fusion_block = model_args.fusion_block
        model.config.fusion_block_lr = training_args.fusion_block_lr


    if model_args.use_memory_block:
        rank0_print(f"[memory] Using memory block")
        model.get_model().initialize_memory_block(model_args=model_args, fsdp=training_args.fsdp)
        dual_memory = model.get_dual_memory()
        if dual_memory is not None:
            target_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
            dual_memory.to(dtype=target_dtype, device=training_args.device)
            rank0_print(f"Dual memory initialized with target_dtype: {target_dtype}")
            model.config.memory_block_lr = training_args.memory_block_lr
    model.config.world_position_embedding_type = model_args.world_position_embedding_type


    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            model.config.tune_spatial_tower = training_args.tune_spatial_tower = model_args.tune_spatial_tower
            model.config.tune_fusion_block = training_args.tune_fusion_block = model_args.tune_fusion_block
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler or model_args.tune_fusion_block or model_args.tune_spatial_tower:
                model.requires_grad_(False)
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if 'lora_' in name:
                        param.requires_grad = True
            if model_args.tune_spatial_tower:
                for p in model.get_spatial_tower().parameters():
                    p.requires_grad = True
            if model_args.tune_fusion_block:
                for p in model.get_fusion_block().parameters():
                    p.requires_grad = True
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)
            if "spatial_tower" in tunable_parts:
                for p in model.get_spatial_tower().parameters():
                    p.requires_grad = True
            if "fusion_block" in tunable_parts:
                for p in model.get_fusion_block().parameters():
                    p.requires_grad = True

        if model_args.world_position_embedding_type is not None:
            for name, param in model.named_parameters():
                if "world_position_embedding" in name:
                    param.requires_grad_(True)
                if "adaptive_pe_mask" in name:
                    rank0_print(f"[adaptive_pe_mask] Training parameter: {name} !!!!!")
                    param.requires_grad_(True)
        if model.config.spatial_tower == 'pi3_camera':
            for name, p in model.get_spatial_tower().named_parameters():
                if "cam_proj" in name:
                    rank0_print(f"Training camer prijector!!!!!")
                    p.requires_grad = True
        
        if model.config.spatial_tower == "pi3_vaga":
            vaga_modules = (
                "local_view_projection",
                "local_geometry_fusion",
                "global_view_projection",
            )
            for name, parameter in model.get_spatial_tower().named_parameters():
                if any(module_name in name for module_name in vaga_modules):
                    rank0_print(f"[VAGA] Training parameter: {name}")
                    parameter.requires_grad_(True)

        if model.config.spatial_tower == 'pi3_camera_pe_local2global':
            for name, p in model.get_spatial_tower().named_parameters():
                if any(keyword in name for keyword in [
                    "cam_proj", 
                    "cam_proj_global", 
                ]):
                    rank0_print(f"[pi3_camera_pe_local2global] -> Training parameter: {name} !!!!!")
                    p.requires_grad = True
        
        if model_args.use_memory_block:
            dual_memory = model.get_dual_memory()
            if dual_memory is not None:
                for name, p in dual_memory.named_parameters():
                    rank0_print(f"[dual_memory] -> Training parameter: {name} !!!!!")
                    p.requires_grad = True
        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
