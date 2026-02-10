"""
torchrun --nproc_per_node=1 linear_fusion.py \
  --pose_ids 0,1,2 \
  --timestep 200,150,50 \
  --blockname mid_0,mid_0,mid_2 \
  --epochs 5 --batch_size 10 --lr 5e-5 --classes 8 \
  --save_csv "eval_fusion_lp_simple_concat.csv" \
  --fusion "simple_concat"

"""

import os
import csv
import math
import argparse
import pickle
from itertools import product
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import random
from collections import defaultdict
from sklearn.model_selection import KFold

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------- Gating Fusion ---------------- #

def exists(x):
    return x is not None


class PerLabelGatingNet(nn.Module):
    
    def __init__(self, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )

    def forward(self, p):
        # p: (B,3,K)
        B, M, K = p.shape
        w = torch.empty(B, M, K, device=p.device)
        for k in range(K):
            logits_k = self.net(p[:, :, k])        # (B,3)
            w[:, :, k] = F.softmax(logits_k, dim=-1)
        return w


class MultiLabelGatingFusion(nn.Module):
    
    def __init__(self, num_labels, hidden=128, dropout=0.1, use_stacking=True):
        super().__init__()
        self.gate = PerLabelGatingNet(hidden=hidden, dropout=dropout)
        self.use_stacking = use_stacking
        if use_stacking:
            self.meta = nn.Sequential(
                nn.Linear(num_labels, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, num_labels)
            )

    def forward(self, logits, return_gate=False):
        
        # Step 1: per-label gating
        w = self.gate(logits)                 # (B,3,K)
        z = (w * logits).sum(dim=1)  # (B,K)

        # Step 2: stacking MLP (optional)
        if self.use_stacking:
            z = self.meta(z)
        if return_gate:
            return z, w  
        return z

class SinglePoseHead(nn.Module):
    def __init__(self, in_dim, num_classes, use_mlp=False):
        super().__init__()
        self.head = LinearOrMLP(in_dim, num_classes, use_mlp=use_mlp)
    def forward(self, x):
        return self.head(x)


def run_kfold_gating_fusion(
    caches_train, fused_train, fused_test,
    num_classes, epochs_base=10, epochs_gate=40, batch_size=256, lr=1e-3, lr_gate=1e-3, use_mlp=False, device="cuda",
    use_stacking=True, n_splits=5
):
    accid  = fused_train["accid"]
    labels = fused_train["labels"]
    N   = labels.shape[0]
    accid_np  = np.array(accid)
    unique_pids = np.array(sorted(list(set(accid_np))))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    logits_oof = torch.zeros(N, 3, num_classes)

    F1 = caches_train[0]["features"].shape[1]
    F2 = caches_train[1]["features"].shape[1]
    F3 = caches_train[2]["features"].shape[1]

    def make_loader(fused, bs, shuffle):
        return data.DataLoader(FusedTensorDataset(fused, fusion="linear_concat"), batch_size=bs, shuffle=shuffle, num_workers=4, worker_init_fn=seed_worker, generator=g)

    for fold, (train_pid_idx, val_pid_idx) in enumerate(kf.split(unique_pids)):
        print0(f"[Gating] Fold {fold+1}/{n_splits}")
        train_pid_set = set(unique_pids[train_pid_idx])
        val_pid_set   = set(unique_pids[val_pid_idx])
        tr_idx = np.where(np.isin(accid_np, list(train_pid_set)))[0]
        va_idx = np.where(np.isin(accid_np, list(val_pid_set)))[0]
        def index_fused(fused, idx):
            out = {}
            for k, v in fused.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v[idx]                      
                else:
                    out[k] = [v[i] for i in idx.tolist()]  
            return out

        tr_fused = index_fused(fused_train, tr_idx)
        va_fused = index_fused(fused_train, va_idx)

        headA = SinglePoseHead(F1, num_classes, use_mlp=use_mlp).to(device)
        headB = SinglePoseHead(F2, num_classes, use_mlp=use_mlp).to(device)
        headC = SinglePoseHead(F3, num_classes, use_mlp=use_mlp).to(device)

        optA = torch.optim.Adam(headA.parameters(), lr=lr)
        optB = torch.optim.Adam(headB.parameters(), lr=lr)
        optC = torch.optim.Adam(headC.parameters(), lr=lr)
        crit = nn.BCEWithLogitsLoss()

        tr_loader = make_loader(tr_fused, batch_size, True)
        for e in range(epochs_base):
            headA.train(); headB.train(); headC.train()
            for xb, yb, _, _ in tr_loader:
                xa, xb_, xc = xb
                yb = yb.to(device).float()
                # A
                optA.zero_grad()
                la = headA(xa.to(device))
                lossA = crit(la, yb)
                lossA.backward(); optA.step()
                # B
                optB.zero_grad()
                lb = headB(xb_.to(device))
                lossB = crit(lb, yb)
                lossB.backward(); optB.step()
                # C
                optC.zero_grad()
                lc = headC(xc.to(device))
                lossC = crit(lc, yb)
                lossC.backward(); optC.step()

        va_loader = make_loader(va_fused, batch_size, False)
        all_pa, all_pb, all_pc = [], [], []
        headA.eval(); headB.eval(); headC.eval()
        with torch.no_grad():
            for xb, yb, _, _ in va_loader:
                xa, xb_, xc = xb
                pa = headA(xa.to(device)).cpu()  # (b,K)
                pb = headB(xb_.to(device)).cpu()
                pc = headC(xc.to(device)).cpu()
                all_pa.append(pa); all_pb.append(pb); all_pc.append(pc)
        pa = torch.cat(all_pa); pb = torch.cat(all_pb); pc = torch.cat(all_pc)   # (n_val,K)
        logits_fold = torch.stack([pa, pb, pc], dim=1)                            # (n_val,3,K)
        logits_oof[va_idx] = logits_fold

    gater = MultiLabelGatingFusion(num_classes, hidden=32, dropout=0.1, use_stacking=use_stacking).to(device)

    opt_g = torch.optim.Adam(gater.parameters(), lr=lr_gate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X = logits_oof.to(device).detach()                 # (N,3,K)
    Y = labels.to(device).float()
    for e in range(epochs_gate):
        gater.train(); opt_g.zero_grad()
        fused_logits = gater(X)                       # (N,K)
        loss = loss_fn(fused_logits, Y)
        loss.backward(); opt_g.step()
        print0(f"[Gating] epoch {e} loss={loss.item():.4f}")

    fullA = SinglePoseHead(F1, num_classes, use_mlp=use_mlp).to(device)
    fullB = SinglePoseHead(F2, num_classes, use_mlp=use_mlp).to(device)
    fullC = SinglePoseHead(F3, num_classes, use_mlp=use_mlp).to(device)
    optA = torch.optim.Adam(fullA.parameters(), lr=lr)
    optB = torch.optim.Adam(fullB.parameters(), lr=lr)
    optC = torch.optim.Adam(fullC.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    full_loader = make_loader(fused_train, batch_size, True)
    for e in range(epochs_base):
        fullA.train(); fullB.train(); fullC.train()
        for xb, yb, _, _ in full_loader:
            xa, xb_, xc = xb
            yb = yb.to(device).float()
            # A
            optA.zero_grad()
            la = fullA(xa.to(device)); lossA = crit(la, yb); lossA.backward(); optA.step()
            # B
            optB.zero_grad()
            lb = fullB(xb_.to(device)); lossB = crit(lb, yb); lossB.backward(); optB.step()
            # C
            optC.zero_grad()
            lc = fullC(xc.to(device)); lossC = crit(lc, yb); lossC.backward(); optC.step()

    test_loader = data.DataLoader(FusedTensorDataset(fused_test, fusion="linear_concat"),
                                  batch_size=batch_size, shuffle=False)
    all_pa, all_pb, all_pc = [], [], []
    fullA.eval(); fullB.eval(); fullC.eval()
    with torch.no_grad():
        for xb, yb, _, _ in test_loader:
            xa, xb_, xc = xb
            pa = fullA(xa.to(device)).cpu()
            pb = fullB(xb_.to(device)).cpu()
            pc = fullC(xc.to(device)).cpu()
            all_pa.append(pa); all_pb.append(pb); all_pc.append(pc)
    pa = torch.cat(all_pa); pb = torch.cat(all_pb); pc = torch.cat(all_pc)
    logits_test = torch.stack([pa, pb, pc], dim=1)       # (M,3,K)

    # with torch.no_grad():
    #     fused_logits = gater(logits_test.to(device))     # (M,K)
    with torch.no_grad():
        fused_logits, gate_weights = gater(logits_test.to(device), return_gate=True)

    os.makedirs("gater_pretrained", exist_ok=True)
    torch.save(gater.state_dict(), "gater_pretrained/gater_pretrained.pt")
    print("[Saved] gater_pretrained.pt")
    return fused_logits.cpu(), fused_test["labels"].cpu(), gater, logits_test, np.array(fused_test["accid"]), gate_weights.cpu()


# ---------------- Utils ---------------- #
def print0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def to_tensor_cpu(x):
    if torch.is_tensor(x):
        return x
    return torch.tensor(x)


def load_pose_cache(pkl_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a PKL file containing: {"features": Tensor[N,F or CxDxHxW], "labels": Tensor[N,K], "accid": Tensor[N]}.
    Accepts both pickle dumped objects and torch.load dumps.
    """
    assert os.path.exists(pkl_path), f"File not found: {pkl_path}"
    try:
        obj = torch.load(pkl_path, map_location="cpu")
    except Exception:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

    # normalize keys
    feats = obj.get("features")
    labels = obj.get("labels")
    accid = obj.get("accid")
    flat = []
    for a in accid:
        if isinstance(a, torch.Tensor):
            if a.ndim == 0:
                flat.append(str(a.item()))
            else:
                flat.extend([str(x.item()) for x in a.reshape(-1)])
        elif isinstance(a, (list, tuple, np.ndarray)):
            for x in a:
                if isinstance(x, torch.Tensor):
                    flat.append(str(x.item()))
                else:
                    flat.append(str(x))
        else:
            flat.append(str(a))

    accid = flat 

    assert feats is not None and labels is not None and accid is not None, (
        f"{pkl_path} must contain keys: features, labels, accid"
    )

    feats = to_tensor_cpu(feats)
    labels = to_tensor_cpu(labels)


    # Ensure contiguous
    return {
        "features": feats.contiguous(),
        "labels": labels.contiguous(),
        "accid": accid,
    }


def build_index_by_accid(accid) -> Dict[str, List[int]]:
    """
    Build an index from acc_id (string) -> list of sample indices.
    """
    idx_map: Dict[str, List[int]] = {}
    for i, a in enumerate(accid):
        idx_map.setdefault(a, []).append(i)
    return idx_map



def fuse_triplets(
    cacheA: Dict[str, torch.Tensor],
    cacheB: Dict[str, torch.Tensor],
    cacheC: Dict[str, torch.Tensor],
    strict_label: bool = True,
    fusion: str = "simple_concat",
) -> Dict[str, torch.Tensor]:
   
    featsA, labelsA, accA = cacheA["features"], cacheA["labels"], cacheA["accid"]
    featsB, labelsB, accB = cacheB["features"], cacheB["labels"], cacheB["accid"]
    featsC, labelsC, accC = cacheC["features"], cacheC["labels"], cacheC["accid"]
    
    print0("Unique accid counts:")
    print0(f"PoseA unique accid: {len(set(accA))}")
    print0(f"PoseB unique accid: {len(set(accB))}")
    print0(f"PoseC unique accid: {len(set(accC))}")

    # Basic checks
    K = labelsA.shape[1]
    assert labelsB.shape[1] == K and labelsC.shape[1] == K, "Label dimension K must match across three poses."

    idxA = build_index_by_accid(accA)
    idxB = build_index_by_accid(accB)
    idxC = build_index_by_accid(accC)

    common_ids = sorted(set(idxA.keys()) & set(idxB.keys()) & set(idxC.keys()))
    print0(f"[Fuse] Patients with all 3 poses: {len(common_ids)}")

    fused_feats: List[torch.Tensor] = []
    fused_labels: List[torch.Tensor] = []
    fused_accid = []
    featsA_list, featsB_list, featsC_list = [], [], []

    skipped_label_mismatch = 0
    total_combos = 0

    for pid in common_ids:
        listA = idxA[pid]
        listB = idxB[pid]
        listC = idxC[pid]

        for ia, ib, ic in product(listA, listB, listC):
            total_combos += 1
            fa, fb, fc = featsA[ia], featsB[ib], featsC[ic]
            la, lb, lc = labelsA[ia], labelsB[ib], labelsC[ic]

            # Enforce label consistency
            if strict_label:
                if not (torch.equal(la, lb) and torch.equal(la, lc)):
                    skipped_label_mismatch += 1
                    continue
                lbl = la
            else:
                # Majority vote (optional)
                lbl = torch.round((la + lb + lc) / 3.0)

            if fusion == "simple_concat":
                fused = torch.cat([fa, fb, fc], dim=-1)
                fused_feats.append(fused.unsqueeze(0))
            else:
                featsA_list.append(fa.unsqueeze(0))
                featsB_list.append(fb.unsqueeze(0))
                featsC_list.append(fc.unsqueeze(0))

            fused_labels.append(lbl.unsqueeze(0))
            fused_accid.append(pid)

    if total_combos > 0:
        ratio = 100.0 * (total_combos - skipped_label_mismatch) / total_combos
    else:
        ratio = 0.0
    print0(f"[Fuse] Total combos: {total_combos}, kept: {total_combos - skipped_label_mismatch} "
           f"({ratio:.2f}%), skipped (label mismatch): {skipped_label_mismatch}")

    if len(fused_labels) == 0:
        raise RuntimeError("No fused samples produced. Check inputs or disable strict_label.")

    fused_labels = torch.cat(fused_labels, dim=0).contiguous()

    if fusion == "simple_concat":
        fused_feats = torch.cat(fused_feats, dim=0).contiguous()
        return {"features": fused_feats, "labels": fused_labels, "accid": fused_accid}
    else:
        return {
            "poseA": torch.cat(featsA_list, dim=0).contiguous(),
            "poseB": torch.cat(featsB_list, dim=0).contiguous(),
            "poseC": torch.cat(featsC_list, dim=0).contiguous(),
            "labels": fused_labels,
            "accid": fused_accid,
        }


class FusedTensorDataset(data.Dataset):
    def __init__(self, fused_cache: Dict[str, torch.Tensor], fusion: str = "simple_concat"):
        self.fusion = fusion
        self.Y = fused_cache["labels"]
        self.A = fused_cache["accid"]
        self.weights = fused_cache.get("weights", None)
        if "features" in fused_cache:   
            self.X = fused_cache["features"]
            self.mode = "simple"
        else:
            self.Xa = fused_cache["poseA"]
            self.Xb = fused_cache["poseB"]
            self.Xc = fused_cache["poseC"]
            self.mode = "multi"
        self.N = len(self.Y)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        w = 1.0
        if self.weights is not None:
            w = self.weights[idx]

        if self.mode == "simple":
            return self.X[idx], self.Y[idx], w, self.A[idx]
        else:
            return (self.Xa[idx], self.Xb[idx], self.Xc[idx]), self.Y[idx], w, self.A[idx]

class CrossAttnFusion(nn.Module):
    """
    Cross-attention fusion for three pose features.
    Each pose attends to the others to produce context-enriched embeddings.
    """
    def __init__(self, dims: Tuple[int, int, int], embed_dim: int = 256, num_heads: int = 4, mode: str = "mean"):
        super().__init__()
        F1, F2, F3 = dims
        self.mode = mode  # "mean" or "concat"

        # Project all poses into same dimension
        self.proj1 = nn.Linear(F1, embed_dim)
        self.proj2 = nn.Linear(F2, embed_dim)
        self.proj3 = nn.Linear(F3, embed_dim)

        # Multihead cross-attention (each pose as query once)
        self.cross12 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross23 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross31 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, fa, fb, fc):
        # [B,F] -> [B,1,E]
        fa = self.proj1(fa).unsqueeze(1)
        fb = self.proj2(fb).unsqueeze(1)
        fc = self.proj3(fc).unsqueeze(1)

        # Each pose attends to another
        attn_a, _ = self.cross12(fa, fb, fb)
        attn_b, _ = self.cross23(fb, fc, fc)
        attn_c, _ = self.cross31(fc, fa, fa)
        # Residual & normalization
        fa = self.norm(fa + attn_a)
        fb = self.norm(fb + attn_b)
        fc = self.norm(fc + attn_c)

        # Pool: mean or concat
        if self.mode == "mean":
            fused = torch.mean(torch.cat([fa, fb, fc], dim=1), dim=1)
        elif self.mode == "concat":
            fused = torch.cat([fa.squeeze(1), fb.squeeze(1), fc.squeeze(1)], dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return fused

class CrossAttnClassifier(nn.Module):
    def __init__(self, dims, embed_dim, num_heads, num_classes, mode="mean", use_mlp=False):
        super().__init__()
        self.fusion = CrossAttnFusion(dims, embed_dim, num_heads, mode)
        out_dim = embed_dim if mode == "mean" else embed_dim * 3
        self.head = LinearOrMLP(out_dim, num_classes, use_mlp=use_mlp)

    def forward(self, fa, fb, fc):
        fused = self.fusion(fa, fb, fc)
        return self.head(fused)

# ---------------- Model ---------------- #
class LinearOrMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, use_mlp: bool = False, hidden_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        if use_mlp:
            h = max(16, int(in_dim * hidden_ratio))
            self.net = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(h, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.net(x)

class LinearFusion(nn.Module):
    def __init__(self, dims: Tuple[int,int,int], out_dim: int, mode: str = "concat"):
        super().__init__()
        F1, F2, F3 = dims
        self.mode = mode
        self.proj1 = nn.Linear(F1, out_dim)
        self.proj2 = nn.Linear(F2, out_dim)
        self.proj3 = nn.Linear(F3, out_dim)

    def forward(self, fa, fb, fc):
        pa = self.proj1(fa)
        pb = self.proj2(fb)
        pc = self.proj3(fc)
        if self.mode == "add":
            return pa + pb + pc                  # [B, out_dim]
        elif self.mode == "concat":
            return torch.cat([pa, pb, pc], dim=-1)  # [B, out_dim*3]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class FusionClassifier(nn.Module):
    def __init__(self, dims, proj_dim, num_classes, mode="concat", use_mlp=False):
        super().__init__()
        self.fusion = LinearFusion(dims, proj_dim, mode)
        out_dim = proj_dim * 3 if mode == "concat" else proj_dim
        self.head = LinearOrMLP(out_dim, num_classes, use_mlp=use_mlp)

    def forward(self, fa, fb, fc):
        fused = self.fusion(fa, fb, fc)
        return self.head(fused)

# ---------------- Evaluation ---------------- #

def patient_level_reduce(accids, preds, labels, reduce="mean"):
    patient_to_probs = {}
    patient_to_label = {}

    for a, p, y in zip(accids, preds, labels):
        if a not in patient_to_probs:
            patient_to_probs[a] = []
            patient_to_label[a] = y
        patient_to_probs[a].append(p)

    agg_preds = []
    agg_labels = []

    for pid in patient_to_probs:
        if reduce == "mean":
            agg_pred = np.mean(patient_to_probs[pid], axis=0)
        elif reduce == "max":
            agg_pred = np.max(patient_to_probs[pid], axis=0)
        else:
            agg_pred = np.median(patient_to_probs[pid], axis=0)

        agg_preds.append(agg_pred)
        agg_labels.append(patient_to_label[pid])

    return np.array(agg_preds), np.array(agg_labels)



def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, acc_id) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    ## patient level
    agg_probs, agg_labels = patient_level_reduce(acc_id, probs, labels_np)
    preds = (agg_probs > 0.5).astype(np.float32)

    K = agg_labels.shape[1]
    ap_per_class, acc_per_class, auc_per_class = [], [], []

    for i in range(K):
        try:
            ap = average_precision_score(agg_labels[:, i], agg_probs[:, i])
        except ValueError:
            ap = np.nan
        ap_per_class.append(ap)

        acc = accuracy_score(agg_labels[:, i], preds[:, i])
        acc_per_class.append(acc)

        try:
            auc = roc_auc_score(agg_labels[:, i], agg_probs[:, i])
        except ValueError:
            auc = np.nan
        auc_per_class.append(auc)

    mean_ap = np.nanmean(ap_per_class)
    mean_auc = np.nanmean(auc_per_class)

    cp = precision_score(agg_labels, preds, average='macro', zero_division=0)
    cr = recall_score(agg_labels, preds, average='macro', zero_division=0)
    cf1 = f1_score(agg_labels, preds, average='macro', zero_division=0)
    op = precision_score(agg_labels, preds, average='micro', zero_division=0)
    or_ = recall_score(agg_labels, preds, average='micro', zero_division=0)
    of1 = f1_score(agg_labels, preds, average='micro', zero_division=0)

    return {
        "APs": ap_per_class,
        "Accs": acc_per_class,
        "AUCs": auc_per_class,
        "mAUCROC": float(mean_auc),
        "mAP": float(mean_ap),
        "CP": float(cp),
        "CR": float(cr),
        "CF1": float(cf1),
        "OP": float(op),
        "OR": float(or_),
        "OF1": float(of1),
    }


def save_metrics_csv(metrics: Dict[str, float], save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ap = metrics["APs"]
    auc = metrics["AUCs"]
    header = ["ACL", "PCL", "MM", "LM", "MCL", "LCL", "PD", "EFFU", "auc_ACL", "auc_PCL", "auc_MM", "auc_LM", "auc_MCL", "auc_LCL", "auc_PD", "auc_EFFU", "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1"]
    row = ap + auc + [metrics["mAP"], metrics["CP"], metrics["CR"], metrics["CF1"], metrics["OP"], metrics["OR"], metrics["OF1"]]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

def save_fused_to_pkl(fused_dict, out_path):

    with open(out_path, "wb") as f:
        pickle.dump(fused_dict, f)
    print(f"[Save] Fused MRI saved to {out_path}, keys: {list(fused_dict.keys())}")


def export_gate_weights_csv(accid_test, gate_weights, save_path="gating_weights.csv"):
    """
    accid_test: (M,)
    gate_weights: (M,3,K)
    """

    M, _, K = gate_weights.shape
    gate_np = gate_weights.numpy()           # (M,3,K)
    gate_np = gate_np.transpose(0, 2, 1)     # (M,K,3) 
    flat = gate_np.reshape(M, K * 3)         # (M, 3*K)

    columns = []
    for k in range(K):
        columns += [f"w_ax_{k}", f"w_sag_{k}", f"w_cor_{k}"]
    df = pd.DataFrame(flat, columns=columns)
    df.insert(0, "accid", accid_test)

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] gating weights exported to: {save_path}")


# ---------------- Training ---------------- #
def train_one_split(
    train_fused: Dict[str, torch.Tensor],
    test_fused: Dict[str, torch.Tensor],
    num_classes: int,
    epochs: int,
    epochs_gate: int,
    batch_size: int,
    lr: float,
    lr_gate: float,
    use_mlp: bool,
    save_csv: str,
    use_stacking: bool,
    fusion: str,                  
    caches_train=None             
):
    device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() and dist.is_initialized() else ("cuda" if torch.cuda.is_available() else "cpu"))

    if fusion == "simple_concat":
        in_dim = train_fused["features"].shape[1]
        model = LinearOrMLP(in_dim, num_classes, use_mlp=use_mlp).to(device)

    elif fusion in ["linear_concat", "linear_add"]:
        F1, F2, F3 = caches_train[0]["features"].shape[1], caches_train[1]["features"].shape[1], caches_train[2]["features"].shape[1]
        proj_dim = 512
        fusion_mode = "concat" if fusion == "linear_concat" else "add"
        model = FusionClassifier((F1, F2, F3), proj_dim, num_classes, fusion_mode, use_mlp).to(device)
    elif fusion == "crossattn":
        F1, F2, F3 = caches_train[0]["features"].shape[1], caches_train[1]["features"].shape[1], caches_train[2]["features"].shape[1]
        embed_dim = 512
        num_heads = 4
        fusion_mode = "concat"  
        model = CrossAttnClassifier((F1, F2, F3), embed_dim, num_heads, num_classes, fusion_mode, use_mlp).to(device)
    elif fusion == "gating":
        logits, labels, gater, probs_test, acc_id, gate_weights = run_kfold_gating_fusion(
            caches_train, train_fused, test_fused,
            num_classes=num_classes, epochs_base=epochs, epochs_gate=epochs_gate, batch_size=batch_size,
            lr=lr, lr_gate=lr_gate, use_mlp=use_mlp, device=device, use_stacking=use_stacking
        )
        export_gate_weights_csv(acc_id, gate_weights, save_path="test_patient_gating_weights.csv")

        label_names = ["ACL", "PCL", "MM", "LM", "MCL", "LCL", "PD", "EFFU"]
        metrics = evaluate_logits(logits, labels, acc_id)
        print0(f"[Gating Fusion Eval] mAUCROC={metrics['mAUCROC']*100:.2f}, mAP={metrics['mAP']*100:.2f}, CF1={metrics['CF1']*100:.2f}, OF1={metrics['OF1']*100:.2f}")
        if is_main():
            save_metrics_csv(metrics, save_csv)
        return

    # DDP (optional)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    train_ds = FusedTensorDataset(train_fused)
    test_ds = FusedTensorDataset(test_fused)

    train_loader = data.DataLoader(
        FusedTensorDataset(train_fused, fusion),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g             
    )

    test_loader = data.DataLoader(
        FusedTensorDataset(test_fused, fusion),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )


    for e in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb, wb, _ in train_loader:
            optimizer.zero_grad()
            yb = yb.to(device).float()
            wb = wb.to(device)
            if fusion == "simple_concat":
                xb = xb.to(device)
                logits = model(xb)
            else:
                xa, xb, xc = [t.to(device) for t in xb]
                logits = model(xa, xb, xc)
            crit = nn.BCEWithLogitsLoss(reduction="none") 
            loss_mat = crit(logits, yb)                        # (B, K)
            loss_per_sample = loss_mat.mean(dim=1)             # (B)
            loss = (loss_per_sample * wb).sum() / wb.sum() 
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        scheduler.step()
        print0(f"[Epoch {e:03d}] train_loss={total_loss/len(train_loader.dataset):.6f}")

    # Evaluation
    model.eval()
    all_logits, all_labels = [], []
    all_accid = []
    with torch.no_grad():
        for xb, yb, _ , accid in test_loader:
            yb = yb.to(device).float()
            if fusion == "simple_concat":
                xb = xb.to(device)
                logits = model(xb)
            else:
                xa, xb, xc = [t.to(device) for t in xb]
                logits = model(xa, xb, xc)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
            all_accid.extend(accid)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logits = logits.cpu()
    labels = labels.cpu()
    

    metrics = evaluate_logits(logits, labels, acc_id=all_accid)
    print0(f"[Eval] mAUCROC={metrics['mAUCROC']*100:.2f}, mAP={metrics['mAP']*100:.2f}, CF1={metrics['CF1']*100:.2f}, OF1={metrics['OF1']*100:.2f}")
    if is_main():
        save_metrics_csv(metrics, save_csv)


def parse_three_paths(arg: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in arg.split(",")]
    assert len(parts) == 3, "Expect exactly three comma-separated paths."
    return parts[0], parts[1], parts[2]

def parse_broadcast_or_list(arg: str, n: int, caster):
    """
    Parse a command-line string that may be a single value or a comma-separated list.
    - If single value: broadcast to length n.
    - If comma-separated: length must equal n.
    caster: callable to cast each token (e.g., int or str)
    """
    tokens = [t.strip() for t in str(arg).split(",")]
    if len(tokens) == 1:
        return [caster(tokens[0])] * n
    if len(tokens) != n:
        raise ValueError(f"Expected {n} values but got {len(tokens)} for '{arg}'.")
    return [caster(t) for t in tokens]

def auto_build_paths(pose_ids: List[int], timesteps: List[int], blocknames: List[str]):
    assert len(pose_ids) == len(timesteps) == len(blocknames), \
        "pose_ids, timesteps, blocknames must have the same length."
    train_paths, test_paths = [], []
    for pid, ts, bn in zip(pose_ids, timesteps, blocknames):
        train_paths.append(f"feature_cache_pooling_ft/pose{pid}_time[{ts}]_name['{bn}']_pooled_train.pkl")
        test_paths.append(f"feature_cache_pooling_ft/pose{pid}_time[{ts}]_name['{bn}']_pooled_valid.pkl")
        # train_paths.append(f"feature_cache_pooling_lp/pose{pid}_time{ts}_name{bn}_pooled_train.pkl")
        # test_paths.append(f"feature_cache_pooling_lp/pose{pid}_time{ts}_name{bn}_pooled_valid.pkl")
    return train_paths, test_paths



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_ids", type=str, required=True,
                        help="Comma-separated pose IDs, e.g. 0,1,2")
    # NOTE: change to str so we can accept lists "30,40,20" or single "30"
    parser.add_argument("--timestep", type=str, required=True,
                        help="Either a single timestep or comma-separated list aligned with pose_ids.")
    parser.add_argument("--blockname", type=str, required=True,
                        help="Either a single blockname or comma-separated list aligned with pose_ids.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epochs_gate", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_gate", type=float, default=1e-3)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--use_mlp", action="store_true")
    parser.add_argument("--use_stacking", action="store_true")
    parser.add_argument("--strict_label", action="store_true", default=True)
    parser.add_argument("--save_csv", type=str, default="eval_fused.csv")
    parser.add_argument("--fusion", type=str, default="simple_concat",
                    choices=["simple_concat", "linear_concat", "linear_add", "crossattn", "gating"],
                    help="Fusion method for 3 poses.")
    parser.add_argument("--seed", type=int, default=42)

    opt = parser.parse_args()
    seed = opt.seed
    set_seed(seed)
    if dist.is_initialized():
        seed_tensor = torch.tensor([seed], device=torch.device(f"cuda:{opt.local_rank}"))
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
        set_seed(seed)
    global g
    g = torch.Generator()
    g.manual_seed(seed)

    # === Auto Build Paths === #
    pose_ids = [int(x.strip()) for x in opt.pose_ids.split(",")]
    timesteps = parse_broadcast_or_list(opt.timestep, len(pose_ids), int)
    blocknames = parse_broadcast_or_list(opt.blockname, len(pose_ids), str)

    train_paths, test_paths = auto_build_paths(pose_ids, timesteps, blocknames)

    print0("[Auto] Train PKLs:", train_paths)
    print0("[Auto] Test  PKLs:", test_paths)

    # === Load and Fuse === #
    caches_train = [load_pose_cache(p) for p in train_paths]
    caches_test = [load_pose_cache(p) for p in test_paths]


    fused_train = fuse_triplets(*caches_train, strict_label=opt.strict_label, fusion=opt.fusion)
    fused_test = fuse_triplets(*caches_test, strict_label=opt.strict_label, fusion=opt.fusion)

    ### for MRI + EHR Fusion
    save_fused_to_pkl(fused_train, "saved_mri_features_train_knee.pkl")
    save_fused_to_pkl(fused_test, "saved_mri_features_test_knee.pkl")
    
    from collections import Counter
    cnt = Counter(fused_train["accid"])
    weights = torch.tensor([1.0 / cnt[a] for a in fused_train["accid"]], dtype=torch.float32)
    fused_train["weights"] = weights


    # === Train and Eval === #
    train_one_split(
        fused_train, fused_test,
        num_classes=opt.classes,
        epochs=opt.epochs,
        epochs_gate=opt.epochs_gate,
        batch_size=opt.batch_size,
        lr=opt.lr,
        lr_gate=opt.lr_gate,
        use_stacking=opt.use_stacking,
        use_mlp=opt.use_mlp,
        save_csv=opt.save_csv,
        fusion=opt.fusion,              
        caches_train=caches_train       
    )

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
