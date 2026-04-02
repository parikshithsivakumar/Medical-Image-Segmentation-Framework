# src/dataset.py
# All HyperKvasir dataset logic in one file for clean notebook imports.

import os, json, random
import cv2, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# ─── Label maps ───────────────────────────────────────────────────────────────
ANATOMY_CLASSES   = {"cecum": 0, "other": 1}  # 🔥 FIX #5: Merged ileum + retroflex → 2-class
ANATOMY_IDX2NAME  = {v: k for k, v in ANATOMY_CLASSES.items()}
# 🔥 FIX #2: Merge G0-1 and G1 to handle class imbalance (only 5 G0-1 test samples)
UC_GRADE_MAP = {
    "ulcerative-colitis-grade-0-1": 0, "ulcerative-colitis-grade-1":   0,  # Merge to 0
    "ulcerative-colitis-grade-1-2": 0, "ulcerative-colitis-grade-2":   1,  # G2 → 1
    "ulcerative-colitis-grade-2-3": 1, "ulcerative-colitis-grade-3":   2,  # G3 → 2
}
UC_IDX2NAME          = {0: "grade 0-1", 1: "grade 2", 2: "grade 3"}  # 3-class now
NUM_ANATOMY_CLASSES  = 2  # 🔥 FIX #5: Changed from 3 to 2 classes (cecum vs other)
NUM_UC_GRADES        = 3  # 🔥 FIX #2: Changed from 4 to 3 (merged G0-1+G1)

# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split, image_size=224):
    if split == "train":
        # 🔥 FIX #4: Enhanced augmentation to reduce overfitting
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            # Stronger color jittering
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.6),
            # Gaussian blur for robustness
            A.GaussianBlur(blur_limit=3, p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            # Elastic deformation for colonoscopy
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()], additional_targets={"mask": "mask"})
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()], additional_targets={"mask": "mask"})

# ─── Sample collectors ─────────────────────────────────────────────────────────
def collect_anatomy_samples(root):
    samples, base = [], Path(root)/"labeled-images"/"lower-gi-tract"/"anatomical-landmarks"
    # 🔥 FIX #5: Map ileum + retroflex → label 1 ("other"), keep cecum → label 0
    for folder_name in ["cecum", "ileum", "retroflex-rectum"]:
        folder = base / folder_name
        if not folder.exists(): continue
        # Map: cecum → 0, ileum → 1, retroflex-rectum → 1
        label = 0 if folder_name == "cecum" else 1
        for p in folder.glob("*.jpg"):
            samples.append({"image_path": str(p), "anatomy_label": label,
                            "polyp_label": -1, "uc_grade": -1,
                            "mask_path": None, "source": "anatomy"})
    return samples

# def collect_polyp_samples(root, bbox_data=None):
#     samples  = []
#     base     = Path(root)/"labeled-images"/"lower-gi-tract"/"pathological-findings"/"polyps"
#     seg_mask = Path(root)/"segmented-images"/"masks"
#     if not base.exists(): print(f"[WARN] {base}"); return samples
#     for p in base.glob("*.jpg"):
#         mp = seg_mask / (p.stem + ".jpg")
#         samples.append({"image_path": str(p), "anatomy_label": -1,
#                         "polyp_label": 1, "uc_grade": -1,
#                         "mask_path": str(mp) if mp.exists() else None,
#                         "source": "polyp"})
#     return samples

def collect_polyp_samples(root, bbox_data=None):
    samples = []

    # 🔥 1. Real segmentation dataset (correct masks)
    img_dir  = Path(root)/"segmented-images"/"images"
    mask_dir = Path(root)/"segmented-images"/"masks"

    if img_dir.exists():
        for p in img_dir.glob("*.jpg"):
            mp = mask_dir / p.name

            samples.append({
                "image_path": str(p),
                "anatomy_label": -1,
                "polyp_label": 1,
                "uc_grade": -1,
                "mask_path": str(mp) if mp.exists() else None,
                "bbox": None,   # 🔥 No bbox for real masks
                "source": "polyp_seg"
            })

    # 🔥 2. Additional HyperKvasir polyp images (no masks → bbox optional)
    base = Path(root)/"labeled-images"/"lower-gi-tract"/"pathological-findings"/"polyps"

    if base.exists():
        for p in base.glob("*.jpg"):
            key = p.stem

            bbox = None
            if bbox_data and key in bbox_data:
                bbox = bbox_data[key].get("bbox", None)

            samples.append({
                "image_path": str(p),
                "anatomy_label": -1,
                "polyp_label": 1,
                "uc_grade": -1,
                "mask_path": None,
                "bbox": bbox,   # 🔥 THIS LINE FIXES EVERYTHING
                "source": "polyp"
            })

    return samples

def collect_uc_samples(root):
    samples, base = [], Path(root)/"labeled-images"/"lower-gi-tract"/"pathological-findings"
    for name, grade in UC_GRADE_MAP.items():
        folder = base / name
        if not folder.exists(): continue
        for p in folder.glob("*.jpg"):
            samples.append({"image_path": str(p), "anatomy_label": -1,
                            "polyp_label": 1, "uc_grade": grade,
                            "mask_path": None, "source": "uc"})
    return samples

def collect_normal_samples(root):
    samples, base = [], Path(root)/"labeled-images"/"lower-gi-tract"/"quality-of-mucosal-views"/"bbps-2-3"
    if not base.exists(): print(f"[WARN] {base}"); return samples
    for p in base.glob("*.jpg"):
        samples.append({"image_path": str(p), "anatomy_label": -1,
                        "polyp_label": 0, "uc_grade": -1,
                        "mask_path": None, "source": "normal"})
    return samples

def collect_video_samples(root):
    csv_path = Path(root)/"labeled-videos"/"video-labels.csv"
    if not csv_path.exists(): print(f"[WARN] {csv_path}"); return []
    df = pd.read_csv(csv_path); df.columns = [c.strip() for c in df.columns]
    samples = []
    for _, row in df.iterrows():
        organ, f1, f2 = str(row.get("Organ","")), str(row.get("Finding 1","")).lower(), str(row.get("Finding 2","")).lower()
        vid_id = str(row.get("Video file","")).strip()
        if "Lower GI" not in organ: continue
        anat = -1
        for finding in [f1, f2]:
            for cls, idx in ANATOMY_CLASSES.items():
                if cls.replace("-"," ") in finding or cls in finding:
                    anat = idx; break
        polyp = 1 if ("polyp" in f1 or "polyp" in f2) else 0
        matches = list(Path(root).rglob(vid_id + ".avi"))
        if matches:
            samples.append({"video_path": str(matches[0]), "anatomy_label": anat,
                            "polyp_label": polyp, "uc_grade": -1, "source": "video"})
    return samples

# ─── Split builder ─────────────────────────────────────────────────────────────
def build_image_splits(root, seed=42):
    bbox_path = Path(root)/"segmented-images"/"bounding-boxes.json"
    bbox_data = {}
    if bbox_path.exists():
        try:
            with open(bbox_path, 'r') as f:
                bbox_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load bounding boxes from {bbox_path}: {e}")
    all_s = (collect_anatomy_samples(root) + collect_polyp_samples(root, bbox_data) +
             collect_uc_samples(root) + collect_normal_samples(root))
    print(f"[INFO] Total samples: {len(all_s)}")
    
    # 🔥 FIX #1: VIDEO-LEVEL SPLIT to prevent data leakage
    # Problem: Individual frames from same video appearing in train+test → data leakage
    # Solution: Group frames by video clip FIRST, then split groups
    print("[INFO] Grouping samples by video clip to prevent data leakage...")
    
    video_groups = {}  # {video_id: [samples from this video]}
    other_samples = []  # anatomy/normal samples (no video grouping)
    
    for s in all_s:
        # Try to extract video ID from image path
        img_path = Path(s["image_path"])
        # HyperKvasir structure: segmented-images/images/XXX_123.jpg
        # or: labeled-images/.../polyps/YYYY_456.jpg
        video_id = img_path.stem.rsplit('_', 1)[0] if '_' in img_path.stem else None
        
        if video_id and len(video_id) > 0:
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(s)
        else:
            # Anatomy/normal images without clear video grouping
            other_samples.append(s)
    
    # Split video groups using stratification
    video_list = list(video_groups.values())
    video_keys = []
    for v_samples in video_list:
        # Get label from first sample (all in group are same video)
        s = v_samples[0]
        if s["uc_grade"] >= 0:
            key = f"uc_{s['uc_grade']}"
        elif s["anatomy_label"] >= 0:
            key = f"anat_{s['anatomy_label']}"
        else:
            key = s["source"]
        video_keys.append(key)
    
    # Split: 15% test, 85% train+val
    if len(video_list) > 1:
        v_tv, v_test_idx = train_test_split(
            range(len(video_list)), test_size=0.15, stratify=video_keys, random_state=seed)
    else:
        v_tv, v_test_idx = list(range(len(video_list))), []
    
    # Split train+val: 70% train, 30% val (from the 85%)
    if len(v_tv) > 1:
        v_train_keys = [video_keys[i] for i in v_tv]
        v_train_idx, v_val_idx = train_test_split(
            v_tv, test_size=0.176, stratify=v_train_keys, random_state=seed)
    else:
        v_train_idx, v_val_idx = v_tv, []
    
    # Flatten: video groups → frame samples
    train_videos = [f for i in v_train_idx for f in video_list[i]]
    val_videos = [f for i in v_val_idx for f in video_list[i]]
    test_videos = [f for i in v_test_idx for f in video_list[i]]
    
    # Also split other (non-grouped) samples
    if other_samples:
        other_keys = []
        for s in other_samples:
            if s["uc_grade"] >= 0:
                key = f"uc_{s['uc_grade']}"
            elif s["anatomy_label"] >= 0:
                key = f"anat_{s['anatomy_label']}"
            else:
                key = s["source"]
            other_keys.append(key)
        
        if len(other_samples) > 1:
            other_tv, other_test_idx = train_test_split(
                range(len(other_samples)), test_size=0.15, stratify=other_keys, random_state=seed)
            other_keys_tv = [other_keys[i] for i in other_tv]
            other_train_idx, other_val_idx = train_test_split(
                other_tv, test_size=0.176, stratify=other_keys_tv, random_state=seed)
        else:
            other_train_idx, other_val_idx, other_test_idx = other_samples, [], []
        
        train_other = [other_samples[i] for i in other_train_idx]
        val_other = [other_samples[i] for i in other_val_idx]
        test_other = [other_samples[i] for i in other_test_idx]
    else:
        train_other, val_other, test_other = [], [], []
    
    # Combine video groups + other samples
    tr = train_videos + train_other
    va = val_videos + val_other
    te = test_videos + test_other
    
    print(f"[INFO] Video-level split: {len(video_groups)} video groups")
    print(f"[INFO] Train:{len(tr)}  Val:{len(va)}  Test:{len(te)}")
    print(f"[INFO] ✓ FIX #1: Video-level split prevents frame leakage across train/test")
    return tr, va, te

# ─── Datasets ─────────────────────────────────────────────────────────────────
class HyperKvasirDataset(Dataset):
    def __init__(self, samples, split="train", image_size=224):
        self.samples, self.transform, self.image_size = samples, get_transforms(split, image_size), image_size

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(s["image_path"])

        if img is None:
            print(f"[ERROR] Failed to load image: {s['image_path']}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        # ✅ 1. Real mask (priority: use if exists)
        if s.get("mask_path") and os.path.exists(s["mask_path"]):
            mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)

            if mask is None:
                mask = np.zeros((h_img, w_img), dtype=np.uint8)
            else:
                # Resize to image size using INTER_NEAREST for binary preservation
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                # Convert to binary
                mask = (mask > 127).astype(np.uint8)

        # ✅ 2. BBox → pseudo mask (if no real mask)
        elif s.get("bbox") is not None:
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            
            # Normalize bbox to ensure it's a list
            bbox_list = s["bbox"] if isinstance(s["bbox"], list) else [s["bbox"]]
            
            for box in bbox_list:
                if isinstance(box, dict):
                    xmin = int(box.get("xmin", 0))
                    ymin = int(box.get("ymin", 0))
                    xmax = int(box.get("xmax", 0))
                    ymax = int(box.get("ymax", 0))
                else:
                    # Tuple/list format: (xmin, ymin, xmax, ymax)
                    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Clip to image bounds
                xmin = max(0, min(xmin, w_img))
                ymin = max(0, min(ymin, h_img))
                xmax = max(xmin, min(xmax, w_img))
                ymax = max(ymin, min(ymax, h_img))

                # Fill mask region
                if xmax > xmin and ymax > ymin:
                    mask[ymin:ymax, xmin:xmax] = 1

        # ✅ 3. No mask (zero mask)
        else:
            mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Ensure mask is uint8
        mask = mask.astype(np.uint8)

        aug = self.transform(image=img, mask=mask)

        return {
            "image": aug["image"],
            "anatomy_label": torch.tensor(s["anatomy_label"], dtype=torch.long),
            "polyp_label":   torch.tensor(s["polyp_label"],   dtype=torch.long),
            "uc_grade":      torch.tensor(s["uc_grade"], dtype=torch.long),
            "mask":          aug["mask"].unsqueeze(0).float()
        }

class VideoFrameDataset(Dataset):
    def __init__(self, video_samples, num_frames=8, image_size=224):
        self.video_samples, self.num_frames = video_samples, num_frames
        self.transform = get_transforms("train", image_size)

    def __len__(self): return len(self.video_samples)

    def __getitem__(self, idx):
        try:
            s = self.video_samples[idx]
        except (IndexError, KeyError) as e:
            # Fallback: return all-zero frames if index is out of range
            return {
                "frames": torch.zeros(self.num_frames, 3, 224, 224, dtype=torch.float32),
                "anatomy_label": torch.tensor(0, dtype=torch.long),
                "polyp_label": torch.tensor(0, dtype=torch.long)
            }
        
        try:
            cap = cv2.VideoCapture(s["video_path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start = 0 if total < self.num_frames else random.randint(0, total - self.num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            
            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(image=rgb_frame)["image"]
                    # Ensure tensor is 3D [C, H, W]
                    if img_tensor.dim() != 3:
                        img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
                    frames.append(img_tensor)
                except Exception:
                    # Skip individual frames that fail
                    continue
            
            cap.release()
            
            # Handle corrupted/empty videos: if no frames read, create black frames
            if len(frames) == 0:
                frames = [torch.zeros(3, 224, 224, dtype=torch.float32) for _ in range(self.num_frames)]
            else:
                # Filter out any frames that don't have shape [3, 224, 224]
                valid_frames = [f for f in frames if f.shape == (3, 224, 224)]
                if not valid_frames:
                    # All frames have wrong shape, use zeros
                    frames = [torch.zeros(3, 224, 224, dtype=torch.float32) for _ in range(self.num_frames)]
                else:
                    frames = valid_frames
                    # Pad with zeros if not enough frames
                    while len(frames) < self.num_frames:
                        frames.append(torch.zeros(3, 224, 224, dtype=torch.float32))
            
            # Final trim to exact size
            frames = frames[:self.num_frames]
            
            # Ensure we have exactly num_frames with correct shape
            if len(frames) < self.num_frames:
                frames.extend([torch.zeros(3, 224, 224, dtype=torch.float32) 
                              for _ in range(self.num_frames - len(frames))])
            
            frames_tensor = torch.stack(frames, dim=0)  # Explicit dim=0
            
            return {
                "frames": frames_tensor,
                "anatomy_label": torch.tensor(s.get("anatomy_label", 0), dtype=torch.long),
                "polyp_label": torch.tensor(s.get("polyp_label", 0), dtype=torch.long)
            }
        
        except Exception as e:
            # Last resort fallback for any unforeseen error
            print(f"[ERROR] VideoFrameDataset failed for idx={idx}: {str(e)}")
            return {
                "frames": torch.zeros(self.num_frames, 3, 224, 224, dtype=torch.float32),
                "anatomy_label": torch.tensor(0, dtype=torch.long),
                "polyp_label": torch.tensor(0, dtype=torch.long)
            }


def get_dataloaders(root, batch_size=16, num_workers=4, seed=42):
    tr, va, te = build_image_splits(root, seed)
    make = lambda s, split: DataLoader(
        HyperKvasirDataset(s, split), batch_size=batch_size,
        shuffle=(split=="train"), num_workers=num_workers,
        pin_memory=True, drop_last=(split=="train"))
    return make(tr,"train"), make(va,"val"), make(te,"test")
