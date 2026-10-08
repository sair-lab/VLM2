<div align="center">

# VLM<sup>2</sup>: Vision-Language Memory for Spatial Reasoning

**ECCV 2026**

[Zuntao Liu](https://sairlab.org/zuntaol/)<sup>1</sup> · [Yi Du](https://sairlab.org/yid/)<sup>1</sup> · [Taimeng Fu](https://sairlab.org/taimengf/)<sup>1</sup> · [Shaoshu Su](https://sairlab.org/shaoshus/)<sup>1</sup> · [Cherie Ho](https://cherieho.com/)<sup>2</sup> · [Chen Wang](https://sairlab.org/chenw/)<sup>1</sup>

<sup>1</sup> Spatial AI &amp; Robotics (SAIR) Lab, University at Buffalo, USA<br>
<sup>2</sup> Stanford University, USA

<p align="center">
    <a href="https://arxiv.org/abs/2511.20644" target="_blank">
    <img src="https://img.shields.io/badge/Paper-B31B1B?style=flat&logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="https://sairlab.org/vlm2/" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-0078D4?style=flat" alt="Project Page">
    </a>
    <a href="https://huggingface.co/waynelio/VLM2-VSI-Bench-Qwen2-7B-LoRA" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFB000?style=flat" alt="Model">
    </a>
</p>

</div>

---

VLM<sup>2</sup> combines visual semantics, geometric cues and persistent memory to reason about spatial relationships in monocular RGB videos. It aligns geometry across viewpoints and maintains working and episodic memories to retain information across frames.

This repository provides the VLM2 model, training code and VSI-Bench evaluation pipeline for the released LLaVA-NeXT-Video/Qwen2-7B adapter.

**Quick links:** [Installation](#installation) · [Model weights](#model-weights) · [Evaluation](#vsi-bench-evaluation) · [Training](#training) · [Citation](#citation)

<a name="overview"></a>

## Overview

<p align="center">
  <img src="https://sairlab.org/img/posts/2025-11-25-vlm2/pipeline.png" width="100%" alt="VLM2 architecture: adaptive 3D position injection, viewpoint-aware geometry alignment, semantic-geometric fusion, and dual memory.">
</p>

VLM2 brings together four components:

| Component | Role |
|---|---|
| **Adaptive 3D position injection** | Adds predicted 3D position cues to visual tokens through a learned gate. |
| **Viewpoint-aware geometry alignment** | Enriches geometry tokens with local and global view information from Pi3. |
| **Semantic-geometric fusion** | Uses cross-attention to combine position-aware visual tokens with viewpoint-aware geometry tokens. |
| **Dual memory** | Retrieves recent context from working memory and earlier observations from episodic memory, then combines both streams with a learned gate. |

See the [paper](https://arxiv.org/abs/2511.20644) and [project page](https://sairlab.org/vlm2/) for the method and experiments.

<a name="installation"></a>

## 🔧 Installation

### 1. Get the code

```bash
git clone --recurse-submodules https://github.com/sair-lab/VLM2.git
cd VLM2
```

Initialize Pi3 and apply the VLM2 compatibility patch before installing:

```bash
git submodule update --init --recursive
git -C Pi3 apply ../patches/pi3_vlm2.patch
```

Pi3 is pinned to [`4fbb4d9`](https://github.com/yyfz/Pi3/tree/4fbb4d9d3bdd098266be776f926af4b9ec05326b). The patch preserves the positional embedding and camera-head dtype handling used by VLM2. Apply it once to a clean checkout of this commit; keep it applied during training and evaluation. Pi3 weights are downloaded separately below.

### 2. Create the environment

The reference environment uses **Linux x86_64, Python 3.10, PyTorch 2.5.1 and CUDA 12.1**. Training and evaluation have been run on NVIDIA H200 GPUs.

Run the following commands from the repository root:

```bash
conda create -n vlm2 python=3.10 -y
conda activate vlm2

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[train,eval]"
```

The PyTorch command follows the [official CUDA 12.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251). `pyproject.toml` installs the VLM2 model, Pi3 package and bundled evaluator together. A separate installation of `thinking-in-space` is not needed.

For evaluation only, use `python -m pip install -e ".[eval]"` in the last step. Dependencies include the pinned Transformers revision and a FlashAttention wheel for the reference environment. Git is required for installation; DeepSpeed training may also require a compatible compiler and CUDA toolkit.

<a name="model-weights"></a>

## Model weights

The released model is available on [Hugging Face](https://huggingface.co/waynelio/VLM2-VSI-Bench-Qwen2-7B-LoRA).

| Component | Source | Loading |
|---|---|---|
| VLM2 LoRA and additional trained modules | [`waynelio/VLM2-VSI-Bench-Qwen2-7B-LoRA`](https://huggingface.co/waynelio/VLM2-VSI-Bench-Qwen2-7B-LoRA) | Download below |
| LLaVA base model | [`lmms-lab/LLaVA-NeXT-Video-7B-Qwen2`](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-Qwen2) | Downloaded automatically when loaded |
| Vision encoder | [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384) | Downloaded automatically when loaded |
| Pi3 geometry encoder | [`yyfz233/Pi3`](https://huggingface.co/yyfz233/Pi3) | Download into `Pi3/checkpoints` |

### Download VLM2 and Pi3

From the repository root:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    repo_id="waynelio/VLM2-VSI-Bench-Qwen2-7B-LoRA",
    local_dir="checkpoints/vlm2_vsibench_llava_qwen2_7b_lora",
)

for filename in ("config.json", "model.safetensors"):
    hf_hub_download(
        repo_id="yyfz233/Pi3",
        filename=filename,
        local_dir="Pi3/checkpoints",
    )
PY
```

The VLM2 checkpoint includes **both** `adapter_model.safetensors` and `non_lora_trainables.bin`. The VLM2 loader loads the base model, restores the additional modules, and merges the LoRA adapter. Keep both weight files and the accompanying configuration files together.

<details>
<summary><strong>Optional: download the LLaVA base model in advance</strong></summary>

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2",
    local_dir="checkpoints/llava_next_video_qwen2_7b",
)
PY
```

To use this local copy for evaluation, set `MODEL_BASE` to its absolute path when launching the evaluator:

```bash
MODEL_BASE="$PWD/checkpoints/llava_next_video_qwen2_7b" \
NUM_GPUS_PER_NODE=8 \
bash thinking-in-space/scripts/eval.sh \
    "$PWD/checkpoints/vlm2_vsibench_llava_qwen2_7b_lora"
```

</details>

<a name="vsi-bench-evaluation"></a>

## 📊 VSI-Bench evaluation

### Prepare the benchmark

Evaluation uses the test split of [`nyu-visionx/VSI-Bench`](https://huggingface.co/datasets/nyu-visionx/VSI-Bench). Authenticate with Hugging Face before running the current task configuration:

```bash
huggingface-cli login
```

The evaluator downloads the dataset and extracts its video archives on the first run. Videos are read from:

```text
$HF_HOME/vsibench/<dataset>/<scene_name>.mp4
```

If `HF_HOME` is unset, the root is `~/.cache/huggingface/`. For a custom cache location, set `HF_HOME` to an absolute path **before** logging in and downloading data. Existing manually prepared videos must follow the same directory structure.

### Run evaluation

From the repository root, on a machine or Slurm allocation with eight GPUs:

```bash
NUM_GPUS_PER_NODE=8 \
bash thinking-in-space/scripts/eval.sh \
    "$PWD/checkpoints/vlm2_vsibench_llava_qwen2_7b_lora"
```

Set `NUM_GPUS_PER_NODE` to match your allocated GPU count. Each GPU runs a complete model replica. On Slurm, the script uses the GPUs assigned to the job through `CUDA_VISIBLE_DEVICES`; GPU resources must be allocated before launching it.

Use `OUTPUT_PATH` to choose another output directory. Aggregate results and per-sample predictions are saved after all evaluation processes finish. The final output path is printed in the log.

<a name="training"></a>

## Training

### Prepare your own training data

**Training annotations and processed training videos are not distributed with this repository.** Prepare your own data using the resources and processing workflow from [VLM-3R](https://github.com/VITA-Group/VLM-3R):

- [Data processing and QA generation guide](https://github.com/VITA-Group/VLM-3R/blob/main/vlm_3r_data_process/README.md)
- [ScanNet preprocessing walkthrough](https://github.com/VITA-Group/VLM-3R/blob/main/vlm_3r_data_process/src/metadata_generation/ScanNet/scannet.md)
- [Upstream training data resources](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA)

Follow the upstream instructions to obtain the source datasets, prepare videos and metadata, and generate QA annotations for the tasks you want to train on. For tasks without a published generation script, use available upstream annotations or prepare your own. Convert the resulting annotations to the conversation JSON expected by the training loader, with `video` and `conversations` fields.

The supplied manifest, [`scripts/VLM2/all_h_vsibench_data.yaml`](scripts/VLM2/all_h_vsibench_data.yaml), references these locally prepared files:

```text
train_json/vsibench_train/
├── scannet_train.json
├── scannetpp_train.json
├── route_planning_train.json
└── scannet_appearance_order_train.json
```

Edit this manifest to match your own annotations, or set `DATA_YAML` to a custom YAML file. For example:

```yaml
datasets:
  - json_path: /absolute/path/to/your_training_annotations.json
```

Every JSON file referenced by the selected manifest must exist before training. The `video` paths in your annotations are resolved relative to `VIDEO_FOLDER`, preserving their spelling and capitalization. For example:

```text
<VIDEO_FOLDER>/
├── ScanNet/videos/train/scene0191_00.mp4
└── scannetpp/videos/train/39f36da05b.mp4
```

Obtain and process the underlying data under the terms of the source datasets, including [ScanNet](https://www.scan-net.org/) and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/).

### Launch training

From the repository root, inside your GPU allocation:

```bash
export IMAGE_FOLDER=/absolute/path/to/processed_data
export VIDEO_FOLDER="$IMAGE_FOLDER"
export DATA_YAML="$PWD/scripts/VLM2/all_h_vsibench_data.yaml"

NUM_GPUS_PER_NODE=8 \
RUN_NAME=vlm2_vsibench \
bash scripts/VLM2/train_vsibench.sh
```

W&B reporting is disabled by default (`--report_to none`). To enable it, change this argument to `--report_to wandb` in `scripts/VLM2/train_vsibench.sh`, run `wandb login` and set `WANDB_MODE=online`. Optionally set `WANDB_ENTITY` and `WANDB_PROJECT` in your environment.

Checkpoints are written to `work_dirs_ours/llava_video_7b_qwen2_<RUN_NAME>` and training logs to `logs_ours/`. Override `OUTPUT_ROOT` and `LOG_DIR` to choose other locations.

<a name="repository-structure"></a>

## Repository structure

```text
.
├── llava/
│   ├── model/                  # VLM2 encoders, fusion and dual memory
│   └── train/                  # Training entry points and data loading
├── Pi3/                        # Pi3 geometry encoder submodule
├── patches/pi3_vlm2.patch       # VLM2 compatibility changes for pinned Pi3
├── scripts/
│   ├── VLM2/                   # Training script and annotation manifest
│   └── zero2_memory.json       # Default DeepSpeed configuration
├── thinking-in-space/
│   ├── lmms_eval/              # VLM2 adapter and VSI-Bench evaluator
│   └── scripts/eval.sh         # Evaluation launcher
├── train_json/vsibench_train/   # Locally prepared annotations (not distributed)
└── pyproject.toml              # Installation configuration
```

<a name="citation"></a>

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{liu2026vlm2,
  title={Vision-Language Memory for Spatial Reasoning},
  author={Liu, Zuntao and Du, Yi and Fu, Taimeng and Su, Shaoshu and Ho, Cherie and Wang, Chen},
  booktitle={European Conference on Computer Vision},
  pages={297--316},
  year={2026},
  organization={Springer}
}
```

<a name="acknowledgments-and-license"></a>

## Acknowledgments

We thank [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [VLM-3R](https://github.com/VITA-Group/VLM-3R), [Pi3](https://github.com/yyfz/Pi3), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) for their open-source contributions.

<a name="license"></a>

## License

VLM2's original code is licensed under [Apache 2.0](LICENSE). Third-party code retains its own terms; see [NOTICE](NOTICE) and the original license copies in [licenses/](licenses/).

The pinned Pi3 version and some of its incorporated code have non-commercial restrictions; MAREPO-derived code also has restrictive redistribution terms. These components are not relicensed by VLM2's Apache license. External model weights and datasets retain their respective licenses.
