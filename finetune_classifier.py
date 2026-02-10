'''
torchrun --nproc_per_node=1 finetune_classifier.py \
    --pose_id 0 \
    --config configs/config_finetune.yaml \
    --weightfile train_diffusion_pose_2/model.pt \
    --classes 8 --finetune_diffusion \
    --save_pooled

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
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score,accuracy_score, roc_auc_score
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler

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
        for image, label, acc_id in loader:
            image = image.to(device)
            t_tensor = torch.full((image.size(0),), t, dtype=torch.long, device=device)
            feat = diffusion.get_feature(image, t_tensor, name=name, use_amp=use_amp)
            # 5D: (B,C,D,H,W) ；2D: (B,F)
            if feat.dim() == 5:
                feat_dim = feat.shape[1]  # C
            else:
                feat_dim = feat.shape[1]  # F
            return feat_dim, feat.dim()
    raise RuntimeError("Empty dataset for probing feature shape.")


def forward_one_batch(diffusion, clf, images, t:int, name:str, device, use_amp:bool):
    images = images.to(device)
    t_tensor = torch.full((images.size(0),), t, dtype=torch.long, device=device)
    feats = diffusion.get_feature(images, t_tensor, name=name, use_amp=use_amp)
    logits = clf(feats)
    return logits, feats



class SelfAttnPooling3D(nn.Module):
    def __init__(self, in_channels, num_heads=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, feat):
        """
        feat: (B, C, D, H, W)
        """
        B, C, D, H, W = feat.shape
        N = D * H * W

        # ===== Flatten =====
        x = feat.view(B, C, N).permute(0, 2, 1)   # (B, N, C)

        # ===== QKV =====
        Q = self.q_proj(x)   # (B, N, C)
        K = self.k_proj(x)   # (B, N, C)
        V = self.v_proj(x)   # (B, N, C)

        # ===== Multi-head attention =====
        head_dim = C // self.num_heads
        Q = Q.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, h, N, d)
        K = K.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B,h,N,N)
        attn_weights = torch.softmax(attn_scores, dim=-1)                       # (B,h,N,N)

        context = torch.matmul(attn_weights, V)  # (B,h,N,d)
        context = context.transpose(1, 2).contiguous().view(B, N, C)  # (B,N,C)

        pooled = context.mean(dim=1)    # (B, C)
        pooled = self.out_proj(pooled)  # (B, C)

        global_feat = feat.mean(dim=(2,3,4))  # (B, C)
        fused_feat = torch.cat([global_feat, pooled], dim=1)  # (B, 2C)

        return fused_feat

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
    # diffusion.eval()
    print0("Model Loaded!")
    return diffusion

# ---------------- Classifier ---------------- #
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, pooling="none"):
        super().__init__()
        self.pooling = pooling
        self.attn_pool = None
        if pooling == "selfattn":
            self.attn_pool = SelfAttnPooling3D(in_dim)
            in_dim *= 2
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feat):
        # feat: (B, C, D, H, W) or (B, F)
        if self.pooling == "selfattn":
            feat = self.attn_pool(feat)
        return self.fc(feat)



def evaluate(cached, classifiers, device, pose, pooling, strength, save_path="eval_results_knee_merged.csv"):
    results = {}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for key, clf in classifiers.items():
        t, name = key.split("/")
        feats = cached["features"][int(t)][name].to(device)
        labels = cached["labels"].to(device).float()  # shape: (B, 6)

        clf.eval()
        with torch.no_grad():
            logits = clf(feats)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        ap_per_class = []
        acc_per_class = []
        auc_per_class = []
        for i in range(labels.shape[1]):
            try:
                ap = average_precision_score(labels[:, i].cpu().numpy(), probs[:, i].cpu().numpy())
            except ValueError:
                ap = np.nan
            ap_per_class.append(ap)
            acc = accuracy_score(labels[:, i].cpu().numpy(), preds[:, i].cpu().numpy())
            acc_per_class.append(acc)
            try:
                auc = roc_auc_score(labels[:, i].cpu().numpy(), probs[:, i].cpu().numpy())
            except ValueError:
                auc = np.nan
            auc_per_class.append(auc)
        mean_ap = np.nanmean(ap_per_class)
        mean_auc = np.nanmean(auc_per_class)

        cp = precision_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        cr = recall_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        cf1 = f1_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        op = precision_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)
        or_ = recall_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)
        of1 = f1_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)

        results[key] = {
            "APs": ap_per_class,
            "Accs": acc_per_class,
            "AUCs": auc_per_class, 
            "mAP": mean_ap,
            "mAUCROC": mean_auc,
            "CP": cp,
            "CR": cr,
            "CF1": cf1,
            "OP": op,
            "OR": or_,
            "OF1": of1
        }

        os.makedirs("finetune_result_knee", exist_ok=True)
        csv_path = f"finetune_result_knee/eval_result_{pooling}_pose_{pose}_{key.replace('/', '_')}_ft.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["ACL", "PCL", "MM", "LM", "MCL", "LCL", "PD", "EFFU", "auc_ACL", "auc_PCL", "auc_MM", "auc_LM", "auc_MCL", "auc_LCL", "auc_PD", "auc_EFFU", "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1"]
            writer.writerow(header)
            row = ap_per_class + auc_per_class + [mean_ap, cp, cr, cf1, op, or_, of1]
            writer.writerow(row)

        print0(f"[{key}] mAUCROC={mean_auc*100:.2f}, mAP={mean_ap*100:.2f}, CF1={cf1*100:.2f}, OF1={of1*100:.2f}")

    return results


# ---------------- Train Loop ---------------- #
def train(opt):
    device = f"cuda:{opt.local_rank}"

    # ---------- Load Config ---------- #
    with open(opt.config, 'r') as f:
        cfg_dict = yaml.full_load(f)
    print0(cfg_dict)
    cfg = Config(cfg_dict)

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
        Lambda(lambda t: t.permute(2, 0, 1)),
        Lambda(lambda t: t.unsqueeze(0))
    ])
    print0("Initializing datasets for finetune...")
    train_set = NiftiImageLabelDataset(
            csv_path=f'dataset/knee_label_train.csv',
            imagefolder='dataset/knee',
            input_size=256,
            depth_size=16,
            transform=transform
    )
    valid_set = NiftiImageLabelDataset(
        csv_path=f'dataset/knee_label_val.csv',
        imagefolder=f'dataset/knee',
        input_size=256,
        depth_size=16,
        transform=transform
    )
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier

    t0, n0 = time_list[0], name_list[0]
    _feat_dim, _feat_ndim = probe_feature_shape(model, train_set, device, t0, n0, opt.use_amp)
    print0(f"[Probe] feature ndim={_feat_ndim}, dim={_feat_dim}")

    in_dim = _feat_dim * 2 if (opt.pooling == "selfattn" and _feat_ndim == 5) else _feat_dim
    clf = LinearClassifier(_feat_dim, opt.classes, pooling=opt.pooling).to(device)
    clf = torch.nn.parallel.DistributedDataParallel(clf, device_ids=[opt.local_rank])

    for p in model.parameters():
        p.requires_grad = False

    trainable_diff_params = []
    for name, p in model.named_parameters():
        if any(key in name.lower() for key in ["middle", "input"]):
            p.requires_grad = True
            trainable_diff_params.append(p)
            print0(f"  -> trainable diffusion param: {name}")

    clf_params = list(clf.module.fc.parameters())
    if hasattr(clf.module, "attn_pool") and clf.module.attn_pool is not None:
        clf_params += list(clf.module.attn_pool.parameters())

    param_groups = []
    if len(clf_params) > 0:
        param_groups.append({"params": clf_params, "lr": base_lr})
    if len(trainable_diff_params) > 0:
        param_groups.append({"params": trainable_diff_params, "lr": base_lr * 0.05})

    finetune_optimizer = torch.optim.Adam(param_groups)
    finetune_scheduler = CosineAnnealingLR(finetune_optimizer, T_max=epoch)
    bce = nn.BCEWithLogitsLoss()

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
        model.train(); clf.train()
        total_loss = 0.0

        for images, labels, acc_id in train_loader_raw:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            finetune_optimizer.zero_grad(set_to_none=True)
            t_tensor = torch.full((images.size(0),), t0, dtype=torch.long, device=device)

            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                feats = model.get_feature(images, t_tensor, name=n0, use_amp=opt.use_amp)
                logits = clf(feats)
                loss = bce(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(finetune_optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)

        finetune_scheduler.step()
        print0(f"[Finetune][Epoch {e}] Loss: {total_loss / len(train_loader_raw.dataset):.6f}")

    def eval(valid_loader_raw):
        print0(">>> Evaluating end-to-end on VALID set <<<")
        model.eval(); clf.eval()

        pseudo_cached = {"features": {t0: {n0: []}}, "labels": [], "accid": []}
        with torch.no_grad():
            for images, labels, acc_id in valid_loader_raw:
                images = images.to(device, non_blocking=True)
                t_tensor = torch.full((images.size(0),), t0, dtype=torch.long, device=device)
                feats = model.get_feature(images, t_tensor, name=n0, use_amp=opt.use_amp).cpu()
                pseudo_cached["features"][t0][n0].append(feats)
                pseudo_cached["labels"].append(labels.clone())
                # acc_id: list[str]
                if isinstance(acc_id, (list, tuple)):
                    pseudo_cached["accid"].extend(acc_id) 
                else:
                    pseudo_cached["accid"].append(acc_id)

        pseudo_cached["features"][t0][n0] = torch.cat(pseudo_cached["features"][t0][n0], dim=0)
        pseudo_cached["labels"] = torch.cat(pseudo_cached["labels"], dim=0)

        results = evaluate(pseudo_cached, {f"{t0}/{n0}": clf}, device, pose_id, opt.pooling)
    eval(valid_loader_raw)
    
    if getattr(opt, "save_pooled", False):
        print0("[Info] Saving pooled features for train and test...")
        os.makedirs(opt.pooled_dir, exist_ok=True)

        def _extract_and_save_pooled_from_raw(dataset, subset_name, strength=None):
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            model.eval(); clf.eval()
            feat_list, label_list, accid_list = [], [], []
            with torch.no_grad():
                for images, labels, acc_id in tqdm(loader, desc=f"Extract pooled from RAW ({subset_name})"):
                    images = images.to(device, non_blocking=True)
                    t_tensor = torch.full((images.size(0),), t0, dtype=torch.long, device=device)
                    feats = model.get_feature(images, t_tensor, name=n0, use_amp=opt.use_amp)
                    if hasattr(clf.module, "attn_pool") and clf.module.attn_pool is not None:
                        pooled_feat = clf.module.attn_pool(feats.to(device)).cpu()
                    elif feats.dim() == 5:
                        pooled_feat = feats.mean(dim=(2,3,4)).cpu()
                    else:
                        pooled_feat = feats.cpu()
                    feat_list.append(pooled_feat)
                    label_list.append(labels.cpu())
                    if isinstance(acc_id, (list, tuple)):
                        accid_list.extend(acc_id)
                    else:
                        accid_list.append(acc_id)

            pooled_cache = {
                "features": torch.cat(feat_list, dim=0),
                "labels": torch.cat(label_list, dim=0),
                "accid": accid_list,
            }
            pooled_filename = f"pose{pose_id}_time{time_list}_name{name_list}_pooled_{subset_name}.pkl"
            pooled_path = os.path.join(
                opt.pooled_dir,
                pooled_filename
            )
            with open(pooled_path, "wb") as f:
                pickle.dump(pooled_cache, f)
            print0(f"[Cache] Saved pooled features → {pooled_path}")

        _extract_and_save_pooled_from_raw(train_set, "train")
        _extract_and_save_pooled_from_raw(valid_set, "valid")

    dist.destroy_process_group()




# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weightfile", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--pose_id", type=int, default=0)
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--pooling", type=str, default="selfattn")
    parser.add_argument("--save_pooled", action='store_true', help="Save pooling features to pkl")
    parser.add_argument("--finetune_diffusion", action='store_true',
                        help="Finetune diffusion model end-to-end with a small LR")
    parser.add_argument("--pooled_dir", type=str, default="feature_cache_pooling_ft",
                        help="Directory to save pooled features pkl")
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