from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
import random
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image

# Dataloader for diffusion pre-training.
# NOTE: You should adapt this implementation to match the format and organization of your own per-training dataset.


def load_pose_ids_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    pose_dict = dict(zip(df['filename'], df['ID']))
    return pose_dict

class NiftiImagePoseGenerator(Dataset):
    def __init__(self,
            imagefolder: str,
            pose_csv_path: str,
            input_size: int,
            depth_size: int,
            transform=None
        ):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.transform = transform
        pose_dict = load_pose_ids_from_csv(pose_csv_path)
        all_files = sorted(glob(os.path.join(imagefolder, '*.nii.gz')))
        self.inputfiles = []
        self.pose_ids = []
        
        for f in tqdm(all_files, desc="Loading NIfTI files"):
            fname = os.path.basename(f)
            if fname not in pose_dict:
                print(f"[SKIP] {fname} not found in pose CSV.")
                continue
            try:
                img = nib.load(f).get_fdata()
                if img.ndim != 3 or img.shape[2] < depth_size:
                    continue
                self.inputfiles.append(f)
                self.pose_ids.append(pose_dict[fname])
            except Exception as e:
                print(f"[SKIP] Failed to load {f}: {e}")
                continue
    def __len__(self):
        return len(self.inputfiles)


    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img
    

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        pose_id = self.pose_ids[index]
        img = self.read_image(inputfile)

        h, w, d = img.shape

        if d > self.depth_size:
            center = d // 2
            half = self.depth_size // 2
            if self.depth_size % 2 == 0:
                start = center - half
                end = center + half
            else:
                start = center - half
                end = center + half + 1
            start = max(0, start)
            end = min(d, end)
            img = img[:, :, start:end]

        if (h, w) != (self.input_size, self.input_size):
            img = np.stack([
                cv2.resize(img[:, :, i], (self.input_size, self.input_size))
                for i in range(img.shape[2])
            ], axis=2)

        if self.transform is not None:
            img = self.transform(img)

        return {
            'target': img,
            'pose_id': torch.tensor(pose_id, dtype=torch.long)
        }


# Dataloader for abnormalities classification tasks.
# NOTE: You should adapt this implementation to match the format and organization of your own classification dataset.

class NiftiImageLabelDataset(Dataset):
    def __init__(
        self,
        csv_path,              # CSV file path (filename, ID, labels)
        imagefolder,           # Root directory containing NIfTI volumes
        input_size,            # Spatial size (H, W)
        depth_size,            # Number of slices along Z axis (D)
        transform=None         # Optional transform applied to [H, W, D]
    ):

        self.input_size = input_size
        self.depth_size = depth_size
        self.imagefolder = imagefolder
        self.transform = transform

        df = pd.read_csv(csv_path)
        df["filename"] = df["filename"].astype(str)

        # -------- Label definitions --------

        ## FOR KNEE
        self.label_cols = [
            "ACL",
            "PCL",
            "MM",
            "LM",
            "MCL",
            "LCL",
            "PD",
            "EFFU"
        ]

        ## FOR ANKLE
        # self.label_cols = [
        #     "ATFL",
        #     "CFL",
        #     "ATR",
        #     "OLT"
        # ]

        ## FOR SHOULDER
        # self.label_cols = [
        #     "SSP",
        #     "ISP",
        #     "SSC",
        #     "LHBT injury",
        #     "AC",
        #     "GHJ",
        #     "LHBT effusion",
        #     "SASD"
        # ]

        self.samples = []

        print("Loading NIfTI metadata...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            relative_path = row["filename"]

            full_path = os.path.join(imagefolder, relative_path)
            if not os.path.exists(full_path):
                # print(f"[SKIP] File not found: {full_path}")
                continue

            try:
                img = nib.load(full_path).get_fdata()

                # Ensure 3D volume
                if img.ndim != 3:
                    # print(f"[SKIP] Invalid image shape: {full_path}")
                    continue

                # Ensure sufficient number of slices
                if img.shape[2] < self.depth_size:
                    # print(f"[SKIP] Insufficient slices: {full_path}")
                    continue

                label = torch.tensor(
                    row[self.label_cols].values.astype(np.float32)
                )

                self.samples.append({
                    "path": full_path,
                    "label": label,
                    "id": row["acc_id"]
                })

            except Exception as e:
                print(f"[SKIP] Failed to load image: {full_path}, error: {e}")
                continue

        print(f"Loaded {len(self.samples)} image samples.")

    def __len__(self):
        return len(self.samples)

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample["path"]
        label = sample["label"]
        acc_id = sample["id"]

        img = self.read_image(file_path)
        h, w, d = img.shape

        # -------- Center crop along depth axis --------
        if d > self.depth_size:
            center = d // 2
            half = self.depth_size // 2

            if self.depth_size % 2 == 0:
                start = center - half
                end = center + half
            else:
                start = center - half
                end = center + half + 1

            # Boundary clipping
            start = max(0, start)
            end = min(d, end)

            img = img[:, :, start:end]

        # -------- Resize spatial resolution --------
        if (h, w) != (self.input_size, self.input_size):
            img = np.stack([
                cv2.resize(
                    img[:, :, i],
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LINEAR
                )
                for i in range(img.shape[2])
            ], axis=2)

        # -------- Optional transform --------
        if self.transform is not None:
            img = self.transform(img)

        return img, label, acc_id



# Dataloader for anatomical segmentation tasks.
# NOTE: You should adapt this implementation to match the format and organization of your own segmentation dataset.

class NiftiImageMaskDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        nii_root: str,
        mask_root: str,
        input_size: int = 256,
        depth_size: int = 16,
        rot_k: int = 3,
        flip_lr: bool = False,
        transform=None,
        pose_id: int = 2,
        strict_mask: bool = True,   # mask 覆盖不足是否丢弃
        verbose: bool = False,
        min_mask_slices: int = 2
    ):
        self.input_size = input_size
        self.depth_size = depth_size
        self.rot_k = rot_k
        self.flip_lr = flip_lr
        self.transform = transform
        self.pose_id = pose_id
        self.strict_mask = strict_mask
        self.verbose = verbose

        self.nii_root = nii_root
        self.mask_root = mask_root

        df = pd.read_csv(csv_path, dtype=str)
        df["pose_id"] = df["pose_id"].astype(int)
        df = df[df["pose_id"] == pose_id]

        if len(df) == 0:
            raise RuntimeError(f"No samples found for pose_id={pose_id}")

        self.groups = []
        dropped = 0

        # ---------- 预过滤样本 ----------
        for key, g in df.groupby(
            ["replath", "study_uid", "nii_file", "slice_direction"]
        ):
            replath, study_uid, nii_file, slice_direction = key
            nii_path = os.path.join(nii_root, replath, nii_file)

            if not os.path.exists(nii_path):
                dropped += 1
                continue

            try:
                Z = nib.load(nii_path).shape[2]
            except Exception:
                dropped += 1
                continue

            if Z < depth_size:
                dropped += 1
                continue

            # mask 覆盖检查（可选）
            if strict_mask:
                g = g.copy()
                g["mask_idx"] = g["mask_file"].str.split("-").str[0].astype(int)

                covered = set()
                for _, row in g.iterrows():
                    z = self._compute_nii_index(
                        int(row["mask_idx"]), Z, slice_direction
                    )
                    if 0 <= z < Z:
                        covered.add(z)

                if len(covered) < min_mask_slices:
                    dropped += 1
                    continue

            self.groups.append((key, g))

        if len(self.groups) == 0:
            raise RuntimeError("All samples filtered out.")

        if verbose:
            print(
                f"[Dataset] kept {len(self.groups)} samples, "
                f"dropped {dropped}"
            )

    def __len__(self):
        return len(self.groups)

    # ---------- utils ----------

    @staticmethod
    def _compute_nii_index(mask_idx_1based, num_slices, slice_direction):
        if slice_direction.lower() == "decreasing":
            return num_slices - mask_idx_1based
        else:
            return mask_idx_1based - 1

    def _load_mask_png(self, path):
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = mask.astype(np.int64)

        if self.flip_lr:
            mask = np.fliplr(mask)
        if self.rot_k > 0:
            mask = np.rot90(mask, k=self.rot_k)

        mask[(mask < 0) | (mask >= 12)] = -1
        return mask

    # ---------- main ----------

    def __getitem__(self, idx):
        (replath, study_uid, nii_file, slice_direction), g = self.groups[idx]

        # ---- load volume ----
        nii_path = os.path.join(self.nii_root, replath, nii_file)
        vol = nib.load(nii_path).get_fdata()
        H, W, Z = vol.shape

        # ---- safe center window ----
        center = Z // 2
        half = self.depth_size // 2
        start = center - half
        end = start + self.depth_size

        if start < 0:
            start = 0
            end = self.depth_size
        if end > Z:
            end = Z
            start = end - self.depth_size
        if start < 0:
            start = 0

        imgs = vol[:, :, start:end].transpose(2, 0, 1)  # (D_actual, H, W)
        d_actual = imgs.shape[0]

        # ---- depth padding (guarantee D == depth_size) ----
        if d_actual < self.depth_size:
            pad = np.zeros(
                (self.depth_size - d_actual, H, W),
                dtype=imgs.dtype
            )
            imgs = np.concatenate([imgs, pad], axis=0)
        elif d_actual > self.depth_size:
            imgs = imgs[:self.depth_size]

        # ---- masks ----
        masks = np.full(
            (self.depth_size, H, W),
            fill_value=-1,
            dtype=np.int64
        )

        g = g.copy()
        g["mask_idx"] = g["mask_file"].str.split("-").str[0].astype(int)

        for _, row in g.iterrows():
            z = self._compute_nii_index(
                int(row["mask_idx"]), Z, slice_direction
            )
            if start <= z < start + self.depth_size:
                z_local = z - start
                mask_path = os.path.join(
                    self.mask_root, study_uid, row["mask_file"]
                )
                if os.path.exists(mask_path):
                    masks[z_local] = self._load_mask_png(mask_path)

        # ---- resize ----
        if (H, W) != (self.input_size, self.input_size):
            imgs = np.stack([
                cv2.resize(
                    imgs[i],
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LINEAR
                )
                for i in range(self.depth_size)
            ])

            masks = np.stack([
                cv2.resize(
                    masks[i],
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_NEAREST
                )
                for i in range(self.depth_size)
            ])

        # ---- transform ----
        if self.transform is not None:
            imgs = self.transform(imgs)

        masks = torch.from_numpy(masks).long()
        acc_id = f"{replath}_{nii_file}"

        return imgs, masks, acc_id
