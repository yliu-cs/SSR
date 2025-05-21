# 🔍 SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning

[SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning](https://arxiv.org/abs/2505.12448)

[Yang Liu*](https://yliu-cs.github.io), Ming Ma*, Xiaomin Yu*, [Pengxiang Ding](https://dingpx.github.io), [Han Zhao](https://h-zhao1997.github.io), Mingyang Sun, [Siteng Huang](https://kyonhuang.top), [Donglin Wang](https://milab.westlake.edu.cn)

[![Static Badge](https://img.shields.io/badge/arXiv-2505.12448-brown)](https://arxiv.org/abs/2505.12448)
[![Static Badge](https://img.shields.io/badge/Project_Page-SSR-blue)](https://yliu-cs.github.io/SSR)
[![Static Badge](https://img.shields.io/badge/HuggingFace-Model_&_Dataset_&_Benchmark-yellow)](https://huggingface.co/collections/yliu-cs/ssr-682d44496b64e4edd94092bb)

![](figure/teaser.jpg)

Despite impressive advancements in Visual-Language Models (VLMs) for multi-modal tasks, their reliance on RGB inputs limits precise spatial understanding. Existing methods for integrating spatial cues, such as point clouds or depth, either require specialized sensors or fail to effectively exploit depth information for higher-order reasoning. To this end, we propose a novel Spatial Sense and Reasoning method, dubbed SSR, a novel framework that transforms raw depth data into structured, interpretable textual rationales. These textual rationales serve as meaningful intermediate representations to significantly enhance spatial reasoning capabilities. Additionally, we leverage knowledge distillation to compress the generated rationales into compact latent embeddings, which facilitate resource-efficient and plug-and-play integration into existing VLMs without retraining. To enable comprehensive evaluation, we introduce a new dataset named SSR-CoT, a million-scale visual-language reasoning dataset enriched with intermediate spatial reasoning annotations, and present SSRBench, a comprehensive multi-task benchmark. Extensive experiments on multiple benchmarks demonstrate SSR substantially improves depth utilization and enhances spatial reasoning, thereby advancing VLMs toward more human-like multi-modal understanding.

## 📚 SSR-CoT Dataset

We collect the following four datasets:

* LLaVA-CoT [[link](https://github.com/PKU-YuanGroup/LLaVA-CoT)]
* Visual-CoT [[link](https://github.com/deepcs233/Visual-CoT)]
* VoCoT [[link](https://github.com/RupertLuo/VoCoT)]
* SpatialQA [[link](https://github.com/BAAI-DCAI/SpatialBot)]

The SSR-CoT dataset directory tree is shown as below:
```sh
SSR-CoT
|-- ssr-cot.jsonl
|-- LLaVA-CoT-100k
|   |-- CLEVR_v1.0
|   |-- CLEVR_v1.0_d  # contains depth
|   `-- ...
|-- Visual-CoT
|   `-- images
|-- VoCoT
|   `-- images
|-- SpatialQA
`   `-- images
```
`ssr-cot.jsonl` can be downloaded from [HuggingFace](https://huggingface.co/datasets/yliu-cs/SSR-CoT).

## 🏠 Installation

```sh
git clone https://github.com/yliu-cs/SSR.git
conda create -n SSR python=3.11
conda activate SSR
cd SSR

pip install -r requirements.txt
```

## 🚀 Training

```sh
# Training Stage 1
accelerate launch --config_file "scripts/fsdp.yaml" ssr/train/train_reasoning.py

# Training Stage 2
accelerate launch --config_file "scripts/fsdp.yaml" ssr/train/train_vlm.py --lora --llava
```

## 📷 Model Checkpoint

| Model   | URL                                                      | Model   | URL                                                      |
|---------|----------------------------------------------------------|---------|----------------------------------------------------------|
| MIDI 7B | [HuggingFace](https://huggingface.co/yliu-cs/SSR-MIDI-7B)| VLM 7B  | [HuggingFace](https://huggingface.co/yliu-cs/SSR-VLM-7B) |

## 🎯 Inference


Coming Soon ...

## 👁️ SSRBench Benchmark

Coming Soon ...

## ❤️ Acknowledgment

Thanks [Meteor](https://github.com/ByungKwanLee/Meteor), [SpatialBot](https://github.com/BAAI-DCAI/SpatialBot), [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT) for their excellent code implementations, which aided later study and are referenced in this implementation as available source code.

## 📜 Citation

Please cite our paper if you use PiTe in your work:

```bibtex
@article{journals/corr/abs-2505-12448,
  author       = {Yang Liu and Ming Ma and Xiaomin Yu and Pengxiang Ding and Han Zhao and Mingyang Sun and Siteng Huang and Donglin Wang},
  title        = {SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning},
  journal      = {CoRR},
  volume       = {abs/2505.12448},
  year         = {2025},
}
```

