'''
torchrun --nproc_per_node=1 linear_classifier.py \
    --pose_id 2 \
    --config config/config_linear.yaml \
    --weightfile train_diffusion_pose_2/model.pt \
    --classes 8

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
from util_func import Config, print0
from torchvision.transforms import Compose, Lambda
import os
import csv
import pickle
import torch.utils.data as data
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score,accuracy_score, roc_auc_score
import numpy as np
import random
from pooling import GlobalLocalPooling3D, SelfAttnPooling3D

def set_seed(seed, local_rank=0):
    seed = seed + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(cfg, opt):

    model = create_model(
        cfg.input_size,
        cfg.num_channels,
        cfg.num_res_blocks,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.input_size,
        depth_size=cfg.depth_size,
        timesteps=cfg.timesteps,
        loss_type='l2'
    ).cuda()

    checkpoint = torch.load(opt.weightfile, map_location="cpu")
    state_dict = checkpoint["ema"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("denoise_fn.module."):
            new_state_dict[k.replace("denoise_fn.module.", "denoise_fn.")] = v
        else:
            new_state_dict[k] = v
    
    diffusion.load_state_dict(new_state_dict, strict=False)
    diffusion.eval()
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
        elif pooling == "local":
            self.attn_pool = GlobalLocalPooling3D(in_dim)
            in_dim *= 2
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feat):
        # feat: (B, C, D, H, W) or (B, F)
        if self.pooling == "selfattn":
            feat = self.attn_pool(feat, return_attn=False)
        elif self.pooling == "local":
            feat = self.attn_pool(feat)
        elif feat.dim() == 5:
            feat = feat.mean(dim=(2, 3, 4))
        return self.fc(feat)

def _plain_loader(dataset, batch_size):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def extract_or_load_features(model, dataset, batch_size, time_list, name_list, device, pose_id, use_amp, cache_dir, split_type="train"):

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"pose{pose_id}_time{time_list}_name{name_list}_{split_type}.pkl")
    raw_cache = os.path.join(cache_dir, f"pose{pose_id}_raw_{split_type}.pkl")
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        if os.path.exists(cache_file):
            print0(f"[Cache] Loading features ({split_type}) from {cache_file}")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            return cache
        if os.path.exists(raw_cache):
            print0(f"[Cache] Loading RAW images from {raw_cache}")
            with open(raw_cache, "rb") as f:
                raw = pickle.load(f)
            images = raw["images"]
            labels = raw["labels"]
            accids = raw["accids"]     
        else:
            print0(f"[Cache] Extracting features ({split_type}) using model.get_feature...")
            loader = _plain_loader(dataset, batch_size)
            img_list, label_list, accid_list = [], [], []

            for image, label, acc_id in tqdm(loader, desc=f"Load RAW {split_type}"):
                img_list.append(image)
                label_list.append(label)
                accid_list.append(acc_id)

            images = torch.cat(img_list, dim=0)                # (N, C, D, H, W)
            labels = torch.cat(label_list, dim=0)              # (N)
            accids = accid_list              # (N)
            with open(raw_cache, "wb") as f:
                pickle.dump({"images": images.cpu(), "labels": labels.cpu(), "accids": accids}, f)
            print0(f"[Cache] Saved RAW to {raw_cache}")
        
        print0(f"[Cache] Extracting features ({time_list}, {name_list}) from RAW")
        model.eval()
        feat_func = partial(model.get_feature, use_amp=use_amp)
        feature_cache = {t: {name: [] for name in name_list} for t in time_list}
        
        with torch.no_grad():
            batch_size_extract = 4
            num_samples = images.size(0)

            for time in time_list:
                feature_cache[time] = {name: [] for name in name_list}

                for i in tqdm(range(0, num_samples, batch_size_extract), desc=f"Time={time}", leave=False):
                    batch = images[i:i+batch_size_extract].to(device, non_blocking=True)
                    t_tensor = torch.full((batch.shape[0],), time, dtype=torch.long, device=device)
                    for name in name_list:
                        feats = feat_func(batch, t_tensor, name=name)
                        feature_cache[time][name].append(feats.cpu())

                for name in name_list:
                    feature_cache[time][name] = torch.cat(feature_cache[time][name], dim=0)


        cache = {"features": feature_cache, "labels": labels, "accids": accids}
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        print0(f"[Cache] Saved features ({split_type}) to {cache_file}")

    if dist.is_initialized():
        dist.barrier()  

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    return cache


def evaluate(cached, classifiers, device, pose, pooling, save_path="eval_results.csv", batch_size=512):
    results = {}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for key, clf in classifiers.items():
        t, name = key.split("/")
        all_feats = cached["features"][int(t)][name]
        all_labels = cached["labels"].float()  # shape: (B, 6)

        clf.eval()
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(all_feats), batch_size):
                batch_feats = all_feats[i:i+batch_size].to(device)
                batch_logits = clf(batch_feats)
                batch_probs = torch.sigmoid(batch_logits)
                batch_preds = (batch_probs > 0.5).float()
                
                all_probs.append(batch_probs.cpu())
                all_preds.append(batch_preds.cpu())
                del batch_feats, batch_logits, batch_probs, batch_preds
            probs = torch.cat(all_probs, dim=0)
            preds = torch.cat(all_preds, dim=0)
            labels = all_labels

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
        # ---- 保存CSV ----
        csv_path = save_path.replace(".csv", f"{pooling}_pose_{pose}_{key.replace('/', '_')}_knee_merged.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["ACL", "PCL", "MM", "LM", "MCL", "LCL", "PD", "EFFU", "auc_ACL", "auc_PCL", "auc_MM", "auc_LM", "auc_MCL", "auc_LCL", "auc_PD", "auc_EFFU", "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1"]
            writer.writerow(header)
            row = ap_per_class + auc_per_class + [mean_ap, cp, cr, cf1, op, or_, of1]
            writer.writerow(row)

        print0(f"[{key}] mAUCROC={mean_auc*100:.2f}, mAP={mean_ap*100:.2f}, CF1={cf1*100:.2f}, OF1={of1*100:.2f}")

    return results

def save_pooling_weights(classifiers, pose_id, pooling, outdir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    os.makedirs(outdir, exist_ok=True)

    for key, clf in classifiers.items():
        module = clf.module if hasattr(clf, "module") else clf
        pool = getattr(module, "attn_pool", None)
        if pool is None:
            continue

        state = pool.state_dict()
        fname_base = f"pose{pose_id}_{pooling}_{key.replace('/', '_')}_pool"
        pt_path = os.path.join(outdir, f"{fname_base}.pt")
        meta = {
            "pose_id": pose_id,
            "pooling": pooling,
            "key": key,
            "state_dict": state
        }
        torch.save(meta, pt_path)

        shape_lines = []
        for n, p in pool.named_parameters():
            shape_lines.append(f"{n}\t{tuple(p.shape)}")
        with open(os.path.join(outdir, f"{fname_base}_shapes.txt"), "w") as f:
            f.write("\n".join(shape_lines))

        print0(f"[Save] Pooling weights saved → {pt_path}")


# ---------------- Train Loop ---------------- #
def train(opt):
    device = f"cuda:{opt.local_rank}"
    # ---------- Load Config ---------- #
    with open(opt.config, 'r') as f:
        cfg_dict = yaml.full_load(f)
    print0(cfg_dict)
    cfg = Config(cfg_dict)
    model = get_model(cfg, opt)

    epoch = cfg.linear['n_epoch']
    batch_size = cfg.linear['batch_size']
    base_lr = cfg.linear['lrate']
    pose_id = opt.pose_id

    time_list = [cfg.linear['timestep']]
    name_list = [cfg.linear['blockname']]

    # ---------- Dataset ---------- #
    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),                            # (H, W, D)
        Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)),         # normalize
        Lambda(lambda t: (t * 2) - 1),                                         # scale to [-1, 1]
        Lambda(lambda t: t.permute(2, 0, 1)),                                 # (D, H, W)
        Lambda(lambda t: t.unsqueeze(0))                                      # (1, D, H, W)
    ])

    cache_dir = "feature_cache"
    train_raw = os.path.join(cache_dir, f"pose{pose_id}_raw_train.pkl")
    valid_raw = os.path.join(cache_dir, f"pose{pose_id}_raw_valid.pkl")

    train_feat = os.path.join(cache_dir, f"pose{pose_id}_time{time_list}_name{name_list}_train.pkl")
    valid_feat = os.path.join(cache_dir, f"pose{pose_id}_time{time_list}_name{name_list}_valid.pkl")

    has_train_raw = os.path.exists(train_raw)
    has_valid_raw = os.path.exists(valid_raw)
    has_train_feat = os.path.exists(train_feat)
    has_valid_feat = os.path.exists(valid_feat)
    
    if not (has_train_raw and has_valid_raw):
        print0("RAW cache not found → creating dataset & loading NIfTI once")
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
        
    else:
        print0("RAW cache exists → skip dataset creation & NIFTI loading")
        train_set = None
        valid_set = None

    cached = extract_or_load_features(model, train_set, batch_size, time_list, name_list, device, pose_id, opt.use_amp, cache_dir)
    cached_valid = extract_or_load_features(model, valid_set, batch_size, time_list, name_list, device, pose_id, opt.use_amp, cache_dir, "valid")

    # scale lr with DDP
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier

    classifiers = {}
    optimizers = {}
    schedulers = {}
    for t in time_list:
        for name in name_list:
            cached["features"][t][name] = cached["features"][t][name].float()
            cached_valid["features"][t][name] = cached_valid["features"][t][name].float()
            feat_dim = cached["features"][t][name].shape[1]
            print("feat shape:", cached["features"][t][name].shape)
            clf = LinearClassifier(feat_dim, opt.classes, pooling=opt.pooling).to(device)
            clf = torch.nn.parallel.DistributedDataParallel(clf, device_ids=[opt.local_rank])
            optim = torch.optim.Adam(clf.parameters(), lr=base_lr)
            sched = CosineAnnealingLR(optim, T_max=epoch)
            key = f"{t}/{name}"
            classifiers[key] = clf
            optimizers[key] = optim
            schedulers[key] = sched

    g = torch.Generator()
    g.manual_seed(opt.seed + opt.local_rank)
    # ---------- Train Loop ---------- #
    for e in range(epoch):
        total_loss = 0.0
        for key, clf in classifiers.items():
            t, name = key.split("/")
            feats_cpu = cached["features"][int(t)][name].float()
            labels_cpu = cached["labels"].float()
            dataset = torch.utils.data.TensorDataset(feats_cpu, labels_cpu)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(opt.seed + worker_id), generator=g)

            for feats_batch, labels_batch in loader:
                feats_batch = feats_batch.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                optimizers[key].zero_grad()
                logits = clf(feats_batch)
                loss = F.binary_cross_entropy_with_logits(logits, labels_batch.float())
                loss.backward()
                optimizers[key].step()
                total_loss += loss.item()

        for s in schedulers.values():
            s.step()
        print0(f"[Epoch {e}] Loss: {total_loss:.4f}")

    if getattr(opt, "save_pooled", False):
        print0("[Info] Saving pooled features...")
        pooled_dir = "feature_cache_pooling"
        os.makedirs(pooled_dir, exist_ok=True)

        def _save_pooled_subset(cached_data, subset_name):
            for key, clf in classifiers.items():
                t, name = key.split("/")
                feats = cached_data["features"][int(t)][name].to(device)
                labels = cached_data["labels"]
                accid = cached_data["accids"]

                clf.eval()
                with torch.no_grad():
                    if hasattr(clf.module, "attn_pool") and clf.module.attn_pool is not None:
                        pooled_feat = clf.module.attn_pool(feats)
                    elif feats.dim() == 5:
                        pooled_feat = feats.mean(dim=(2, 3, 4))  # fallback: global avg pooling

                pooled_cache = {
                    "features": pooled_feat.cpu(),
                    "labels": labels.cpu(),
                    "accid": accid,
                }
                pooled_filename = f"pose{pose_id}_time{t}_name{name}_pooled_{subset_name}.pkl"
                pooled_path = os.path.join(
                    pooled_dir,
                    pooled_filename
                )
                with open(pooled_path, "wb") as f:
                    pickle.dump(pooled_cache, f)
                print0(f"[Cache] Saved pooled features → {pooled_path}")

        _save_pooled_subset(cached, "train")
        _save_pooled_subset(cached_valid, "valid")
    results = evaluate(cached_valid, classifiers, device, pose_id, opt.pooling)

    if getattr(opt, "save_pool_weights", False):
        save_pooling_weights(classifiers, pose_id, opt.pooling, opt.pool_weight_outdir)

        
    dist.destroy_process_group()



# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weightfile", type=str, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--pose_id", type=int, default=0)
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--pooling", type=str, default="selfattn")
    parser.add_argument("--save_pooled", action='store_true', help="Save pooling features to pkl")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save_pool_weights", action='store_true',
                    help="Save pooling layer weights after training (rank0 only)")
    parser.add_argument("--pool_weight_outdir", type=str, default="pooling_weights",
                        help="Output directory for pooling weights")


    opt = parser.parse_args()
    print0(opt)

    local_rank = opt.local_rank
    set_seed(opt.seed, local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    train(opt)