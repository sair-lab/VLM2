#    Copyright 2023 Haotian Liu
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


import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    if "llava" in model_name.lower() or is_multimodal:
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading LLaVA from base model...")
            if "qwen" in model_name.lower():
                from llava.model.language_model.llava_qwen import LlavaQwenConfig
                del kwargs["device_map"]
                additional_config = {
                    "tie_word_embeddings": False,
                    "use_cache": True,
                    "vocab_size": 152064
                }
                lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                if overwrite_config is not None:
                    overwrite_config.update(additional_config)
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(lora_cfg_pretrained, k, v)
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, attn_implementation=attn_implementation, config=lora_cfg_pretrained, **kwargs)
                else:
                    overwrite_config = additional_config
                    for k, v in overwrite_config.items():
                        setattr(lora_cfg_pretrained, k, v)
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, attn_implementation=attn_implementation, config=lora_cfg_pretrained, **kwargs)

            else:
                from llava.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
                rank0_print("=== Verifying saved weights ===")
                position_keys = [k for k in non_lora_trainables.keys() if "world_position_embedding" in k]
                mask_keys = [k for k in non_lora_trainables.keys() if "adaptive_pe_mask" in k]
                memory_keys = [k for k in non_lora_trainables.keys() if "dual_memory" in k]
                rank0_print(f"Found world_position_embedding keys: {position_keys}")
                rank0_print(f"Found adaptive_pe_mask keys: {mask_keys}")
                rank0_print(f"Found dual_memory keys: {memory_keys}")
                if not position_keys:
                    rank0_print("WARNING: No world_position_embedding weights found in saved model!")
                if not mask_keys:
                    rank0_print("WARNING: No adaptive_pe_mask weights found in saved model!")
                if not memory_keys:
                    rank0_print("WARNING: No dual_memory weights found in saved model!")
                rank0_print("=== End verification ===")
           
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            incompatible = model.load_state_dict(non_lora_trainables, strict=False)
            custom_prefixes = (
                "model.mm_projector.", "model.fusion_block.",
                "model.world_position_embedding.", "model.adaptive_pe_mask.",
                "model.dual_memory.",
            ) + tuple(
                f"model.spatial_tower.spatial_tower.spatial_model.encoder.{head}."
                for head in ("local_view_projection", "local_geometry_fusion", "global_view_projection")
            )
            missing_custom = sorted(
                name for name in incompatible.missing_keys
                if name.startswith(custom_prefixes)
            )
            if missing_custom or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint does not match model implementation. "
                    f"Missing custom weights: {missing_custom}; "
                    f"unexpected non-LoRA weights: {incompatible.unexpected_keys}"
                )

            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()

        elif model_base is not None:
            if (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                llava_cfg = LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=llava_cfg, **kwargs)
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                from llava.model.language_model.llava_qwen import LlavaQwenConfig
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                additional_config = {
                    "tie_word_embeddings": False,
                    "use_cache": True,
                    "vocab_size": 152064
                }
                if overwrite_config is not None:
                    overwrite_config.update(additional_config)
                    cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                    del kwargs["device_map"]
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, attn_implementation=attn_implementation, config=cfg_pretrained, **kwargs)
                    model.to(device="cuda", dtype=torch.float16)
                else:
                    overwrite_config = additional_config
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, attn_implementation=attn_implementation, **kwargs)
            else:
                raise ValueError(f"Model {model_name} not supported")

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")         

            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            rank0_print(f"Loaded LLaVA model: {model_path}")

            if (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if "moe" in model_name.lower() or "A14B" in model_name.lower():
                    raise ValueError("This VLM2 release does not support this language backbone.")

                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig
                    additional_config = {
                        "tie_word_embeddings": False,
                        "use_cache": True,
                        "vocab_size": 151649
                    }
                    del kwargs["device_map"]
                    if overwrite_config is not None:
                        overwrite_config.update(additional_config)
                        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                        model.to(device="cuda", dtype=torch.float16)
                    else:
                        overwrite_config = additional_config
                        model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, attn_implementation=attn_implementation, **kwargs)

            else:
                try:
                    from llava.model.language_model.llava_llama import LlavaConfig

                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if customized_config is None:
                        llava_cfg = LlavaConfig.from_pretrained(model_path)
                        if "v1.5" in model_path.lower():
                            llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                    else:
                        llava_cfg = customized_config

                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                except:
                    raise ValueError(f"Model {model_name} not supported")

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if "llava" in model_name.lower() or is_multimodal:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
