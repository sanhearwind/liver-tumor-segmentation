import os
# =========================================================
# 0. 环境与性能预设
# =========================================================
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import sys
import subprocess
import importlib.util

def ensure_package(pkg_name: str, pip_name: str | None = None):
    if importlib.util.find_spec(pkg_name) is not None: return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg_name])

ensure_package("einops")

import re
import gc
import json
import time
import random
import logging
import traceback
from glob import glob

import requests
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByLabelClassesd,
    RandFlipd, RandRotate90d, RandGaussianNoised, EnsureTyped, AsDiscrete,
    Lambdad
)
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.networks.nets import DynUNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# =========================================================
# 1. 核心超参数 (★ 已修改：降低学习率进行微调)
# =========================================================
PATCH_SIZE = (192, 192, 192)  
PIXDIM = (1.0, 1.0, 1.5)      
FEAT_SIZE = 32
NUM_CLASSES = 2
BEST_LR = 2e-5 # ★ 从 1e-4 降到 2e-5，开启微调模式
MAX_EPOCHS = 100 # 微调不需要 300 轮，100 轮足够了
VAL_INTERVAL = 5
PATIENCE = 15

# 采样与数据增强
TRAIN_CLASS_RATIOS = [1, 2, 4] 
TRAIN_NUM_SAMPLES_PER_CASE = 2 

# 路径管理
ROOT_DIR = os.path.expanduser("~/autodl-tmp/LiTs")
RUN_DIR = os.path.expanduser("~/autodl-tmp/lits_full_train_192")
os.makedirs(RUN_DIR, exist_ok=True)

CKPT_PATH = os.path.join(RUN_DIR, "checkpoint_finetune.pth") # ★ 改个新名字，防止覆盖旧断点
BEST_TUMOR_PATH = os.path.join(RUN_DIR, "best_tumor_model.pth")
BEST_META_PATH = os.path.join(RUN_DIR, "best_tumor_meta.json")
LOG_PATH = os.path.join(RUN_DIR, "train_finetune.log")

# 设备与通知
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Keep notification credentials out of source control. Configure this only
# in the runtime environment when notification is needed.
WECHAT_TOKEN = os.environ.get("AUTODL_WECHAT_TOKEN", "")
AUTO_SHUTDOWN = False

# =========================================================
# 2. 日志与微信通知
# =========================================================
logger = logging.getLogger("LiTS-V10")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_h = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_h.setFormatter(formatter)
    logger.addHandler(file_h)
    stream_h = logging.StreamHandler(sys.__stdout__)
    stream_h.setFormatter(formatter)
    logger.addHandler(stream_h)

def send_wechat(msg: str):
    try:
        clean_msg = str(msg).replace('\n', ' ').replace('\r', ' ')
        clean_msg = re.sub(r'[^\w\s\.\-\:]', '', clean_msg)
        if len(clean_msg) > 20: 
            clean_msg = clean_msg[:17] + "..."
            
        headers = {"Authorization": WECHAT_TOKEN}
        requests.post(
            "https://www.autodl.com/api/v1/wechat/message/send",
            json={"title": "LiTS", "name": clean_msg, "content": "Update"},
            headers=headers, timeout=5
        )
    except Exception as e:
        logger.warning(f"WeChat Send Failed: {e}")

# =========================================================
# 3. 标签转换与损失函数
# =========================================================
def convert_to_subregions(label):
    whole_liver = (label == 1) | (label == 2)
    tumor = (label == 2)
    return torch.cat([whole_liver.float(), tumor.float()], dim=0)

class SubRegionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7, batch=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        if torch.is_tensor(pred) and pred.ndim == 6: 
            pred = torch.unbind(pred, dim=1)
            
        if isinstance(pred, (list, tuple)):
            weights = [1.0 / (1.5**i) for i in range(len(pred))]
            total_loss = 0
            for i, p in enumerate(pred):
                curr_target = torch.nn.functional.interpolate(target, size=p.shape[2:], mode='nearest') if p.shape[2:] != target.shape[2:] else target
                total_loss += weights[i] * (self.tversky(p, curr_target) + self.bce(p, curr_target))
            return total_loss / sum(weights)
        
        return self.tversky(pred, target) + self.bce(pred, target)

# =========================================================
# 4. 数据 Pipeline
# =========================================================
def get_data_loaders():
    all_imgs = sorted(glob(os.path.join(ROOT_DIR, "**", "volume-*.nii"), recursive=True), key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0]))
    all_labs = sorted(glob(os.path.join(ROOT_DIR, "segmentations", "segmentation-*.nii")), key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0]))
    
    img_map = {int(re.findall(r"\d+", os.path.basename(p))[0]): p for p in all_imgs}
    lab_map = {int(re.findall(r"\d+", os.path.basename(p))[0]): p for p in all_labs}
    valid_ids = sorted(set(img_map.keys()) & set(lab_map.keys()))
    
    data = [{"image": img_map[i], "label": lab_map[i]} for i in valid_ids]
    random.seed(42)
    random.shuffle(data)
    
    pre_trans = Compose([
        LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=250, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE), 
        EnsureTyped(keys=["image", "label"]),
    ])
    
    train_aug = Compose([
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, num_classes=3, ratios=TRAIN_CLASS_RATIOS, num_samples=TRAIN_NUM_SAMPLES_PER_CASE),
        Lambdad(keys=["label"], func=convert_to_subregions),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys=["image"], prob=0.15), 
        EnsureTyped(keys=["image", "label"]),
    ])
    
    val_trans = Compose([
        Lambdad(keys=["label"], func=convert_to_subregions),
        EnsureTyped(keys=["image", "label"])
    ])
    
    train_files, val_files = data[:-20], data[-20:]
    logger.info(f"正在加载 CacheDataset: 训练 {len(train_files)} / 验证 {len(val_files)}")
    
    train_ds_base = CacheDataset(train_files, pre_trans, cache_rate=1.0, num_workers=8)
    val_ds_base = CacheDataset(val_files, pre_trans, cache_rate=1.0, num_workers=4)
    
    class AugWrapper(Dataset):
        def __init__(self, ds, trans): self.ds, self.trans = ds, trans
        def __len__(self): return len(self.ds)
        def __getitem__(self, i): return self.trans(self.ds[i])
        
    train_loader = DataLoader(AugWrapper(train_ds_base, train_aug), batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(AugWrapper(val_ds_base, val_trans), batch_size=1, shuffle=False)
    
    return train_loader, val_loader

# =========================================================
# 5. 训练主函数
# =========================================================
def main():
    train_loader, val_loader = get_data_loaders()
    
    strides, kernels = [[1,1,1]], [[3,3,3]]
    curr_size = list(PATCH_SIZE)
    while all(s > 4 for s in curr_size) and len(strides) < 6:
        s = [2 if i >= 8 else 1 for i in curr_size]
        strides.append(s); kernels.append([3,3,3])
        curr_size = [i // j for i, j in zip(curr_size, s)]
        
    model = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
        kernel_size=kernels, strides=strides, upsample_kernel_size=strides[1:],
        filters=[FEAT_SIZE * (2**i) for i in range(len(strides))],
        dropout=0.1, norm_name="instance", deep_supervision=True, res_block=True
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    loss_func = SubRegionLoss()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    
    start_epoch, patience_counter = 0, 0
    # ★ 我们把及格线直接定在 0.57，只有超过这个分数才会被保存
    best_t_dice = 0.57 
    
    # =========================================================
    # ★ 核心修改：暴力加载 0.57 模型机制
    # =========================================================
    if os.path.exists(CKPT_PATH):
        # 如果微调跑了一半意外中断，从这里无缝恢复
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sch"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_t_dice, patience_counter = ckpt["epoch"]+1, ckpt["best"], ckpt["patience"]
        logger.info(f"🔄 检测到微调中断，从 Epoch {start_epoch} 继续冲刺！")
        
    elif os.path.exists(BEST_TUMOR_PATH):
        # 第一次跑这个脚本，强制把 0.57 分的本体抽出来塞进模型里
        ckpt = torch.load(BEST_TUMOR_PATH, map_location=DEVICE)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        logger.info("🚀 成功注入 0.57 最佳本体权重，开启低学习率微调冲刺！")
    else:
        logger.warning("❌ 没找到你的 best_tumor_model.pth 文件，请检查路径！")
    # =========================================================

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            imgs, labs = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_func(model(imgs), labs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    for v_batch in val_loader:
                        v_img, v_lab = v_batch["image"].to(DEVICE), v_batch["label"].to(DEVICE)
                        
                        def predictor(x):
                            y = model(x)
                            if torch.is_tensor(y) and y.ndim == 6:
                                return y[:, 0]
                            elif isinstance(y, (list, tuple)):
                                return y[0]
                            return y
                        
                        out = sliding_window_inference(v_img, PATCH_SIZE, 1, predictor, overlap=0.5)
                        preds = [AsDiscrete(threshold=0.5)(torch.sigmoid(x)) for x in decollate_batch(out)]
                        dice_metric(y_pred=preds, y=decollate_batch(v_lab))
            
            res = dice_metric.aggregate()
            l_d, t_d = res[0].item(), res[1].item()
            logger.info(f"E{epoch+1} | Liver: {l_d:.4f} | Tumor: {t_d:.4f}")
            
            # ★ 只有超过 0.57 才会触发保存
            if t_d > (best_t_dice + 1e-4):
                send_wechat(f"E{epoch+1} L{l_d:.2f} T{t_d:.2f}")
                best_t_dice, patience_counter = t_d, 0
                torch.save(model.state_dict(), BEST_TUMOR_PATH)
                with open(BEST_META_PATH, "w", encoding="utf-8") as f: 
                    json.dump({"epoch": epoch+1, "tumor_dice": t_d, "liver_dice": l_d}, f)
                logger.info(f"★★★ 破纪录了！当前最高 Tumor Dice: {best_t_dice:.4f} ★★★")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE: 
                    logger.info("分数学不动了，触发早停，微调结束。"); break

        torch.save({
            "epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict(),
            "sch": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best": best_t_dice, "patience": patience_counter,
            "config": {"patch": PATCH_SIZE, "feat": FEAT_SIZE}
        }, CKPT_PATH)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
        send_wechat("Train Success")
    except Exception as e:
        error_info = str(e).split(':')[-1].strip()
        send_wechat(f"Err:{error_info[:15]}")
        logger.error(traceback.format_exc())
    finally:
        if AUTO_SHUTDOWN:
            time.sleep(10)
            os.system("/usr/bin/shutdown")
