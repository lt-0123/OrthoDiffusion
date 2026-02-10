'''
torchrun --nproc_per_node=1 finetune_segmentation.py \
    --pose_id 0 --config config/config_segmentation.yaml \
    --weightfile train_diffusion_pose_2/model.pt \
    --num_classes 11 \
    --finetune_diffusion

'''
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from functools import partial
from typing import Dict
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from dataset import NiftiImageLabelDataset
from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model
from util_func import Config, init_seeds, gather_tensor, DataLoaderDDP, print0
import pandas as pd
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
import os
import csv
import pickle
import torch.utils.data as data
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
from dataset import NiftiImageMaskDataset as SegDataset
from typing import Dict, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

SEG_COLORS = np.array([
    [235, 230, 220],
    [255, 0,   0  ],   # 1 red
    [0,   255, 0  ],   # 2 green
    [0,   0,   255],   # 3 blue
    [255, 255, 0  ],   # 4 yellow
    [255, 0,   255],   # 5 magenta
    [0,   255, 255],   # 6 cyan
    [128, 0,   0  ],   # 7 dark red
    [0,   128, 0  ],   # 8 dark green
    [0,   0,   128],   # 9 dark blue
    [255, 128, 0  ],   # 10 orange
], dtype=np.uint8)

def set_seed(seed, local_rank=0):
    seed = seed + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def probe_feature_shape(diffusion, dataset, device, t:int, name:str, use_amp:bool):
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for image, mask, acc_id in loader:
            image = image.to(device)
            t_tensor = torch.full((image.size(0),), t, dtype=torch.long, device=device)
            feat = diffusion.get_feature(image, t_tensor, name=name, use_amp=use_amp)
            # 5D: (B,C,D,H,W) ；2D: (B,F)
            if feat.dim() != 5:
                raise RuntimeError(f"[Probe] Segmentation need shape (B,C,D,H,W)")
            return feat.shape[1], feat.shape[2:], feat.dim()
    raise RuntimeError("Empty dataset for probing feature shape.")

def unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m




def get_model(cfg, opt, device):
    model = create_model(
        cfg.input_size,
        cfg.num_channels,
        cfg.num_res_blocks,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.input_size,
        depth_size=cfg.depth_size,
        timesteps=cfg.timesteps,
        loss_type='l2'
    ).to(device)

    checkpoint = torch.load(opt.weightfile, map_location="cpu")
    state_dict = checkpoint["ema"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("denoise_fn.module."):
            new_state_dict[k.replace("denoise_fn.module.", "denoise_fn.")] = v
        else:
            new_state_dict[k] = v
    
    diffusion.load_state_dict(new_state_dict, strict=False)
    print0("Model Loaded!")
    return diffusion

# ---------------- Seg ---------------- #
class SegHead3D(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_ch, in_ch//2, 4, 2, 1),   # 16→32
            nn.ReLU(),
            ResBlock3D(in_ch//2),
            nn.ConvTranspose3d(in_ch//2, in_ch//4, 4, 2, 1), # 32→64
            nn.ReLU(),
            ResBlock3D(in_ch//4),
            nn.ConvTranspose3d(in_ch//4, in_ch//8, 4, 2, 1), # 64→128
            nn.ReLU(),
            nn.ConvTranspose3d(in_ch//8, in_ch//16, 4, 2, 1),# 128→256
            nn.ReLU(),
            nn.Conv3d(in_ch//16, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)   # (B,C,D,256,256)
    
class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))



# ---------------- Loss / Metrics ---------------- #
def soft_dice_loss(logits, target, ignore_index=-1, eps=1e-6):
    """
    logits : (B, C, D, H, W)
    target : (B, D, H, W), class-id
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    valid = (target != ignore_index)
    target_safe = target.clone()
    target_safe[~valid] = 0

    target_oh = F.one_hot(target_safe, num_classes)\
                  .permute(0, 4, 1, 2, 3)\
                  .float()
    valid = valid.unsqueeze(1).float()  # (B,1,D,H,W)
    probs = probs * valid
    target_oh = target_oh * valid

    dims = (0, 2, 3, 4)
    inter = (probs * target_oh).sum(dims)
    union = probs.sum(dims) + target_oh.sum(dims)

    dice = (2 * inter + eps) / (union + eps)   # (C,)
    fg_mask = torch.ones(num_classes, device=dice.device, dtype=torch.bool)
    fg_mask[0] = False
    return 1.0 - dice[fg_mask].mean()

def label_to_color(mask, colors=SEG_COLORS):
    """
    mask: (H, W) int64
    return: (H, W, 3) uint8
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(colors))
    color_mask[valid] = colors[mask[valid]]
    return color_mask


def visualize_segmentation(
    image,      # (D,H,W) or (H,W)
    gt,
    pred,
    save_dir,
    case_id,
    slice_idx=None,
    alpha=0.4
):

    os.makedirs(save_dir, exist_ok=True)

    if image.dim() == 3:
        image = image.cpu().numpy()
    else:
        image = image.squeeze(0).cpu().numpy()

    gt   = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    D = image.shape[0]

    img_slice  = image[slice_idx]
    gt_slice   = gt[slice_idx]
    pred_slice = pred[slice_idx]

    gt_color   = label_to_color(gt_slice)
    pred_color = label_to_color(pred_slice)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_slice, cmap="gray")
    axs[0].set_title("Image")

    axs[1].imshow(img_slice, cmap="gray")
    axs[1].imshow(gt_color, alpha=alpha)
    axs[1].set_title("GT (11 classes)")

    axs[2].imshow(img_slice, cmap="gray")
    axs[2].imshow(pred_color, alpha=alpha)
    axs[2].set_title("Prediction (11 classes)")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{case_id}_slice{slice_idx}.png"),
        dpi=150
    )
    plt.close()


@torch.no_grad()
def validate(
    model,
    seg_head,
    loader,
    device,
    t0,
    n0,
    num_classes,
    sample_num,
    ignore_index=-1,
    pose_id=1,
    save_vis=False,
    max_vis_cases=30
):
    model.eval()
    seg_head.eval()
    vis_cnt = 0

    dice_sum_per_class = defaultdict(float)
    dice_cnt_per_class = defaultdict(int)

    for images, masks, acc_id in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()

        t = torch.full((images.size(0),), t0, dtype=torch.long, device=device)
        feats  = model.get_feature(images, t, name=n0)
        logits = seg_head(feats)
        pred   = torch.argmax(logits, dim=1)  # (B,D,H,W)

        if save_vis and dist.get_rank() == 0:
            for b in range(images.size(0)):
                if vis_cnt >= max_vis_cases:
                    break
                D = images.size(2)  # (B,1,D,H,W)

                for slice_idx in range(D):
                    visualize_segmentation(
                        image=images[b, 0],      # (D,H,W)
                        gt=masks[b],
                        pred=pred[b],
                        save_dir=f"finetune_seg/vis_pose{pose_id}/{acc_id[b]}",
                        case_id=f"{acc_id[b]}",
                        slice_idx=slice_idx
                    )
                vis_cnt += 1

        valid = (masks != ignore_index)

        c_start = 0

        for c in range(c_start, num_classes):
            gt_c = (masks == c) & valid
            if gt_c.sum().item() == 0:
                continue  

            pred_c = (pred == c) & valid
            inter = (pred_c & gt_c).sum().item()
            den   = pred_c.sum().item() + gt_c.sum().item()

            dice = (2 * inter + 1e-6) / (den + 1e-6)

            dice_sum_per_class[c] += dice
            dice_cnt_per_class[c] += 1

    # ---- summary ----
    dice_per_class = {}
    for c in dice_sum_per_class:
        dice_per_class[c] = dice_sum_per_class[c] / dice_cnt_per_class[c]

    mean_dice = sum(dice_per_class.values()) / max(1, len(dice_per_class))
    rows = []
    for c, d in sorted(dice_per_class.items()):
        rows.append({
            "class_id": c,
            "dice": d
        })

    rows.append({
        "class_id": "mean",
        "dice": mean_dice
    })

    df = pd.DataFrame(rows)

    # ---- save ----
    save_path = f"finetune_seg/eval_dice_per_class_{t0}_{n0}_{pose_id}.csv"
    df.to_csv(save_path, index=False)

    return mean_dice, dice_per_class



# ---------------- Train Loop ---------------- #
def train(opt):
    device = f"cuda:{opt.local_rank}"

    # ---------- Load Config ---------- #
    with open(opt.config, 'r') as f:
        cfg_dict = yaml.full_load(f)
    print0(cfg_dict)
    cfg = Config(cfg_dict)

    # only finetune
    if not opt.finetune_diffusion:
        print0("[Warn] You asked to only consider finetune. Forcing --finetune_diffusion=True.")
        opt.finetune_diffusion = True

    # Build model on local device
    model = get_model(cfg, opt, device)

    epoch = cfg.linear['n_epoch']
    batch_size = cfg.linear['batch_size']
    base_lr = cfg.linear['lrate']
    pose_id = opt.pose_id

    time_list = [cfg.linear['timestep']]
    name_list = [cfg.linear['blockname']]

    # ---------- Dataset ---------- #
    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.unsqueeze(0))
    ])
    print0("Initializing datasets for finetune...")
    train_set = SegDataset(
        csv_path=f'dataset/train_segmentation_{pose_id}.csv',
        nii_root="dataset/kneesegmentdata",
        mask_root="dataset/kneesegmentdata_mask",
        transform=transform,
        pose_id=pose_id
    )
    valid_set = SegDataset(
        csv_path=f'dataset/valid_segmentation_{pose_id}.csv',
        nii_root="dataset/kneesegmentdata",
        mask_root="dataset/kneesegmentdata_mask",
        transform=transform,
        pose_id=pose_id
    )

    # scale lr with DDP
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier

    t0, n0 = time_list[0], name_list[0]
    feat_c, feat_spatial, feat_ndim = probe_feature_shape(model, train_set, device, t0, n0, opt.use_amp)
    print0(f"[Probe] feature ndim={feat_ndim}, C={feat_c}, spatial={feat_spatial}")

    seg_head = SegHead3D(in_ch=feat_c, num_classes=opt.num_classes).to(device)
    seg_head = torch.nn.parallel.DistributedDataParallel(seg_head, device_ids=[opt.local_rank])

    for p in model.parameters():
        p.requires_grad = False

    trainable_diff_params = []
    for name, p in model.named_parameters():
        if any(key in name.lower() for key in ["middle", "input"]):
            p.requires_grad = True
            trainable_diff_params.append(p)
            print0(f"  -> trainable diffusion param: {name}")

    seg_params = list(unwrap_ddp(seg_head).parameters())

    param_groups = [
        {"params": seg_params, "lr": base_lr},
    ]
    if len(trainable_diff_params) > 0:
        param_groups.append({"params": trainable_diff_params, "lr": base_lr * 0.05})

    optimizer = torch.optim.Adam(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    
    def _worker_init_fn(worker_id):
        wseed = opt.seed + worker_id
        np.random.seed(wseed)
        random.seed(wseed)
        
    g = torch.Generator()
    g.manual_seed(opt.seed + opt.local_rank)
    
    # Dataloaders
    train_loader_raw = data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, worker_init_fn=_worker_init_fn, generator=g)
    valid_loader_raw = data.DataLoader(
        valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=4, pin_memory=True, worker_init_fn=_worker_init_fn, generator=g)
    
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
  
    for e in range(epoch):
        train_sampler.set_epoch(e)
        model.train(); seg_head.train()
        total_loss = 0.0
        total_ce   = 0.0
        total_dice = 0.0
        total_n    = 0

        for step, (images, masks, acc_id) in enumerate(train_loader_raw):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()
            bad = (masks != -1) & ((masks < 0) | (masks >= opt.num_classes))
            if bad.any():
                bad_vals = torch.unique(masks[bad]).detach().cpu().tolist()
                print0(f"[FATAL] invalid labels in batch step={step}, acc_id={acc_id}")
                print0(f"[FATAL] invalid values (unique) = {bad_vals[:50]}")
                print0(f"[FATAL] masks min/max = {masks.min().item()} / {masks.max().item()}")
                raise RuntimeError("Invalid class id in masks")
            bs = images.size(0)
            total_n += bs
            
            optimizer.zero_grad(set_to_none=True)
            t_tensor = torch.full((images.size(0),), t0, dtype=torch.long, device=device)

            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                feats = model.get_feature(images, t_tensor, name=n0, use_amp=opt.use_amp)
                logits = seg_head(feats)
                loss_ce   = ce_loss(logits, masks)
                loss_dice = soft_dice_loss(logits, masks, ignore_index=-1)
                loss      = 0.2 * loss_ce + loss_dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * bs
            total_ce   += loss_ce.item() * bs
            total_dice += loss_dice.item() * bs

        scheduler.step()
        avg_loss = total_loss / max(1, total_n)
        avg_ce   = total_ce   / max(1, total_n)
        avg_dice = total_dice / max(1, total_n)

        print0(
            f"[Train][Epoch {e}] "
            f"avg_loss={avg_loss:.6f} "
            f"(ce={avg_ce:.6f}, dice={avg_dice:.6f}) "
            f"lr={scheduler.get_last_lr()[0]:.3e}"
        )
    print0(">>> Evaluating end-to-end on VALID set <<<")
    mean_dice, dice_per_class = validate(
        model,
        seg_head,
        valid_loader_raw,
        device,
        t0,
        n0,
        num_classes=opt.num_classes,
        sample_num=sample_num,
        ignore_index=-1,
        pose_id=pose_id
    )

    print0(f"dicescore={mean_dice:.4f}")



# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weightfile", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--pose_id", type=int, default=0)
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--finetune_diffusion", action='store_true',
                        help="Finetune diffusion model end-to-end with a small LR")
    # add after other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    opt = parser.parse_args()
    print0(opt)
    local_rank = opt.local_rank
    
    set_seed(opt.seed, local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank
    train(opt)