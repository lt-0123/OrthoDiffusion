
# OrthoDiffusion

**OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation**

This repository provides the official PyTorch implementation of **OrthoDiffusion**, a diffusion-based foundation model for musculoskeletal MRI analysis, supporting:

* ✅ Multi-plane MRI representation learning
* ✅ Anatomical segmentation
* ✅ Multi-label disease diagnosis
* ✅ Label-efficient training
* ✅ Cross-anatomy transfer (knee → ankle / shoulder)
* ✅ MRI + EHR multimodal fusion

---

## Overview

OrthoDiffusion consists of **three orientation-specific 3D diffusion backbones**:

* Sagittal
* Coronal
* Axial

Each diffusion model is pretrained in a **self-supervised** manner on large-scale unlabeled knee MRI data. Intermediate denoising features are extracted and reused for downstream tasks:

* **Classification** via pooling + fusion heads
* **Segmentation** via lightweight decoder heads

Multi-plane fusion is achieved by:

* Feature-level concatenation (default)
* Linear fusion
* Cross-attention
* **MPAE (Multi-plane Adaptive Expert)** for interpretability

---

## ⚙️ Installation

### Environment

```bash
conda create -n orthodiff python=3.10
conda activate orthodiff

pip install -r requirements.txt
```

Key dependencies:

* PyTorch ≥ 2.0
* einops
* numpy
* scikit-learn
* nibabel

---

## 🧠 Diffusion Pretraining (Using `med-ddpm`)

### Overview

We adopt the **med-ddpm** repository for self-supervised 3D diffusion pretraining on MRI volumes.

**Repo:** [https://github.com/mobaidoctor/med-ddpm](https://github.com/mobaidoctor/med-ddpm)

```bash
torchrun --nproc_per_node=1 train.py \
    --pose_id 1 \
    --results_folder "results_pose_1" > log.log
```

---

## 🩺 Disease Diagnosis

### Stage I (Pooling)

#### Linear Probing

Freeze diffusion backbone:

```bash
torchrun --nproc_per_node=1 linear_classifier.py \
    --pose_id 2 \
    --config configs/config_linear.yaml \
    --weightfile train_diffusion_pose_2/model.pt \
    --classes 8
```

#### Fine-tuning

```bash
torchrun --nproc_per_node=1 finetune_classifier.py \
    --pose_id 0 \
    --config configs/config_finetune.yaml \
    --weightfile train_diffusion_pose_0/model.pt \
    --classes 8 --finetune_diffusion \
    --save_pooled
```

### Stage II (Fusion)

```bash
torchrun --nproc_per_node=1 linear_fusion.py \
  --pose_ids 0,1,2 \
  --timestep 200,150,50 \
  --blockname mid_0,mid_0,mid_2 \
  --epochs 5 --batch_size 10 --lr 5e-5 --classes 8 \
  --save_csv "eval_fusion_lp_simple_concat.csv" \
  --fusion "simple_concat"
```

### Other (MRI + EHR fusion)

```bash
python fusion_info.py
```

Supports:

* SAP pooling
* Multi-plane fusion
* MRI + EHR logits fusion
* Cross-joint transfer

---

## 🦴 Anatomical Segmentation

```bash
torchrun --nproc_per_node=1 finetune_segmentation.py \
    --pose_id 1 --config configs/config_segmentation.yaml \
    --weightfile train_diffusion_pose_1/model.pt \
    --num_classes 11 \
    --finetune_diffusion
```

Features:

* Encoder + bottleneck fine-tuned
* Lightweight decoder
* Dice + CE loss


---

## Dataset

Due to privacy restrictions, MRI data is **not publicly released**.

Partial access can be requested from the corresponding authors.

---

## Citation

If you use this code, please cite:

```
@article{orthodiffusion2025,
  title={OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation},
  author={Lan, Tian and Xu, Lei and Yuan, Zimu et al.},
  journal={},
  year={2025}
}
```


## Contact

For questions:

* Dingyu Wang — [wang_dingyu@pku.edu.cn](mailto:wang_dingyu@pku.edu.cn)

