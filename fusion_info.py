import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import random
from collections import Counter
from typing import Dict
import pickle

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    header = ["ACL", "PCL", "MM", "LM", "MCL", "LCL", "PD", "EFFU",
              "auc_ACL", "auc_PCL", "auc_MM", "auc_LM", "auc_MCL", "auc_LCL", "auc_PD", "auc_EFFU",
              "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1"]
    row = ap + auc + [metrics["mAP"], metrics["CP"], metrics["CR"], metrics["CF1"], metrics["OP"], metrics["OR"], metrics["OF1"]]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
        

class MRITabularDataset(Dataset):
    def __init__(
        self,
        mri_feat,         
        labels,               
        df,                   
        acc_ids,              
        event_col="event",
        cont_cols=("age","height","weight"),
        sex_col="sex",
        patient_type_col="patient_type",
        scaler=None,
        event_stoi=None,
        fit_on_this=False,
        event_embed_dim=8
    ):
        # ---- MRI feature ----
        if isinstance(mri_feat, torch.Tensor):
            mri_feat = mri_feat.detach().cpu().numpy()
        self.mri = mri_feat.astype("float32")

        # ---- Labels ----
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        self.labels = labels.astype("float32")

        # ---- acc_ids (list or np array) ----
        if isinstance(acc_ids, torch.Tensor):
            acc_ids = acc_ids.detach().cpu().numpy()
        self.acc_ids = acc_ids
        N = len(self.acc_ids)

        cnt = Counter(self.acc_ids)
        # weight = 1 / #samples_of_patient
        self.weights = np.array([1.0 / cnt[a] for a in self.acc_ids], dtype="float32")

        df_mri = pd.DataFrame({"acc_id": self.acc_ids})
        df = df_mri.merge(df, on="acc_id", how="left") 

        if event_stoi is None:
            uniq = df[event_col].fillna("UNK").astype(str).unique().tolist()
            if "UNK" not in uniq:
                uniq.append("UNK")
            uniq = sorted(uniq)
            self.event_stoi = {s:i for i,s in enumerate(uniq)}
        else:
            self.event_stoi = event_stoi

        self.event_ids = df[event_col].fillna("UNK").astype(str).map(
            lambda x: self.event_stoi.get(x, self.event_stoi["UNK"])
        ).values.astype("int64")

        self.num_events = len(self.event_stoi)
        self.event_embed_dim = event_embed_dim

        exist_cont = [c for c in cont_cols if c in df.columns]
        if len(exist_cont) > 0:
            Xc = df[exist_cont].fillna(0).values.astype("float32")
            if scaler is None:
                scaler = StandardScaler()
            if fit_on_this:
                Xc = scaler.fit_transform(Xc)
            else:
                Xc = scaler.transform(Xc)
        else:
            Xc = np.zeros((len(df), 0), dtype="float32")

        self.X_cont = Xc
        self.scaler = scaler

        if sex_col in df.columns:
            s = df[sex_col].fillna("UNK").astype(str)
            cats = ["0","1","UNK"]
            mapper = {c:i for i,c in enumerate(cats)}
            idx = s.map(lambda x:
                        mapper["1"] if "1" in x else
                        (mapper["0"] if "0" in x else mapper["UNK"])
                        ).values
            self.X_sex = np.zeros((len(df), len(cats)), dtype="float32")
            self.X_sex[np.arange(len(df)), idx] = 1.0
        else:
            self.X_sex = np.zeros((len(df), 0), dtype="float32")


        if patient_type_col in df.columns:
            s = df[patient_type_col].fillna("UNK").astype(str)
            cats = ["0","1","UNK"]
            mapper = {c:i for i,c in enumerate(cats)}
            idx = s.map(lambda x:
                        mapper["1"] if "1" in x else
                        (mapper["0"] if "0" in x else mapper["UNK"])
                        ).values
            self.X_patient_type = np.zeros((len(df), len(cats)), dtype="float32")
            self.X_patient_type[np.arange(len(df)), idx] = 1.0
        else:
            self.X_patient_type = np.zeros((len(df), 0), dtype="float32")


    def __len__(self):
        return len(self.mri)

    def __getitem__(self, idx):
        x_mri = torch.from_numpy(self.mri[idx])     # [512]
        x_cont = torch.from_numpy(self.X_cont[idx])
        x_sex = torch.from_numpy(self.X_sex[idx])
        x_patient_type = torch.from_numpy(self.X_patient_type[idx])
        event_id = torch.tensor(self.event_ids[idx], dtype=torch.long)
        y = torch.from_numpy(self.labels[idx])
        w = torch.tensor(self.weights[idx], dtype=torch.float32)
        acc_id = self.acc_ids[idx]
        return x_mri, x_cont, x_sex, x_patient_type, event_id, y, w, acc_id


class LateFusionModel(nn.Module):
    def __init__(
        self,
        mri_dim=1536,
        cont_dim=3,
        sex_dim=3,
        patient_dim=3,
        num_events=20,
        event_emb_dim=8,
        hidden_tab=64,
        num_labels=8
    ):
        super().__init__()


        self.mri_head = nn.Linear(mri_dim, num_labels)


        self.event_embed = nn.Embedding(num_events, event_emb_dim)

        tab_in = cont_dim + sex_dim + patient_dim + event_emb_dim
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_in, hidden_tab),
            nn.ReLU(),
            nn.LayerNorm(hidden_tab),
            nn.Dropout(0.2)
        )
        self.tab_head = nn.Linear(hidden_tab, num_labels)

        self.alpha_param = nn.Parameter(torch.zeros(num_labels))
        # alpha = sigmoid(alpha_param)

    def forward(self, x_mri, x_cont, x_sex, x_patient, event_id):
        # ----- MRI branch -----
        mri_logit = self.mri_head(x_mri)       # [B, num_labels]

        # ----- Tabular branch -----
        e = self.event_embed(event_id)         # [B, emb]
        tab_x = torch.cat([x_cont, x_sex, x_patient, e], dim=-1)
        tab_h = self.tab_proj(tab_x)           # [B, hidden_tab]
        tab_logit = self.tab_head(tab_h)

        # ----- Learnable late fusion -----
        alpha = torch.sigmoid(self.alpha_param)    # scalar ∈ (0,1)
        fusion = alpha * mri_logit + (1 - alpha) * tab_logit

        return fusion, mri_logit, tab_logit, alpha

class TabularOnlyNet(nn.Module):
    def __init__(self, cont_dim, sex_dim, patient_dim, num_events, event_embed_dim, num_labels, hidden=64):
        super().__init__()
        self.event_embed = nn.Embedding(num_events, event_embed_dim)

        in_dim = cont_dim + sex_dim + patient_dim + event_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.2)
        )
        self.out = nn.Linear(hidden, num_labels)
    def forward(self, x_cont, x_sex, x_patient_type, event_id):
        e = self.event_embed(event_id)
        x = torch.cat([x_cont, x_sex, x_patient_type, e], dim=-1)
        h = self.mlp(x)
        return self.out(h)

def run_training():
    
    set_seed(2026)
    MRI_PKL = "saved_mri_features_train_knee.pkl"   # 训练集
    MRI_PKL_TEST = "saved_mri_features_test_knee.pkl"
    CSV_FILE = "dataset/EHR_info.csv"
    LABEL_COLS = ["ACL", "PCL", "MM", "LM", "MCL", "LCL", "PD", "EFFU"]  # 你的多标签列名
    SAVE_CSV = "results_ehr_info.csv"
    EPOCHS = 10
    LR = 1e-5
    
    df = pd.read_csv(CSV_FILE)

    with open(MRI_PKL, "rb") as f:
        train_pkl = pickle.load(f)

    with open(MRI_PKL_TEST, "rb") as f:
        test_pkl = pickle.load(f)

    mri_tr = train_pkl["features"]       # [N,512]
    y_tr = train_pkl["labels"]           # [N,K]
    acc_tr = train_pkl["accid"]        # list[str]

    mri_te = test_pkl["features"]
    y_te = test_pkl["labels"]
    acc_te = test_pkl["accid"]

    ds_tr_tmp = MRITabularDataset(
        mri_tr, y_tr, df, acc_tr,
        event_col="event",
        cont_cols=("age","height","weight"),
        sex_col="sex",
        patient_type_col="patient_type",
        scaler=None,
        event_stoi=None,
        fit_on_this=True,   
        event_embed_dim=8
    )
    scaler = ds_tr_tmp.scaler
    event_stoi = ds_tr_tmp.event_stoi

    ds_tr = MRITabularDataset(
        mri_tr, y_tr, df, acc_tr,
        event_col="event",
        cont_cols=("age","height","weight"),
        sex_col="sex",
        patient_type_col="patient_type",
        scaler=scaler,
        event_stoi=event_stoi,
        fit_on_this=False,
        event_embed_dim=8
    )

    ds_te = MRITabularDataset(
        mri_te, y_te, df, acc_te,
        event_col="event",
        cont_cols=("age","height","weight"),
        sex_col="sex",
        patient_type_col="patient_type",
        scaler=scaler,
        event_stoi=event_stoi,
        fit_on_this=False,
        event_embed_dim=8
    )

    train_loader = DataLoader(ds_tr, batch_size=10, shuffle=True)
    test_loader = DataLoader(ds_te, batch_size=10)


    mri_dim = mri_tr.shape[1]
    cont_dim = ds_tr.X_cont.shape[1]
    sex_dim = ds_tr.X_sex.shape[1]
    patient_type_dim = ds_tr.X_patient_type.shape[1]
    num_events = len(event_stoi)
    num_labels = y_tr.shape[1]  
    # model = TabularOnlyNet(cont_dim, sex_dim, patient_type_dim, num_events, event_embed_dim=8, num_labels=num_labels)
    model = LateFusionModel(num_events=num_events)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(reduction="none") 


    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_mri, x_cont, x_sex, x_patient_type, event_id, y, w, _ in train_loader:
            fusion, mri_logit, tab_logit, alpha = model(x_mri, x_cont, x_sex, x_patient_type, event_id)
            # logit = model(x_cont, x_sex, x_patient_type, event_id)
            loss_mat = crit(fusion, y)                # (B,K)
            # loss_mat = crit(logit, y)
            loss_per_sample = loss_mat.mean(dim=1)    # (B)
            loss = (loss_per_sample * w).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        print(f"Epoch {epoch+1} loss = {total_loss/len(ds_tr):.4f}")

        all_logits, all_labels, all_accid = [], [], []
        model.eval()
        with torch.no_grad():
            for x_mri, x_cont, x_sex, x_patient_type, event_id, y, _, accid in test_loader:
                fusion, _, _, _ = model(x_mri, x_cont, x_sex, x_patient_type, event_id)
                # logit = model(x_cont, x_sex, x_patient_type, event_id)
                all_logits.append(fusion.cpu())
                # all_logits.append(logit.cpu())
                all_labels.append(y.cpu())
                all_accid.extend(accid)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        logits = logits.cpu()
        labels = labels.cpu()
        metrics = evaluate_logits(logits, labels, acc_id=all_accid)
        print(f"[Eval] mAUCROC={metrics['mAUCROC']*100:.2f}, mAP={metrics['mAP']*100:.2f}, CF1={metrics['CF1']*100:.2f}, OF1={metrics['OF1']*100:.2f}")
    save_metrics_csv(metrics, SAVE_CSV)
    print("Saved:", SAVE_CSV)

if __name__ == "__main__":
    run_training()
