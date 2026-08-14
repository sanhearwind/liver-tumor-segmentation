import os

# 必须放在 import torch 之前，否则 CUDA allocator / multiprocessing 配置可能不生效
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# 重要：multiprocessing 会在 TMPDIR 下创建 AF_UNIX socket。
# 路径过长会触发 OSError: AF_UNIX path too long，所以这里必须使用短路径。
MP_TMPDIR = "/tmp/lits_mp"
os.makedirs(MP_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = MP_TMPDIR
os.environ["TEMP"] = MP_TMPDIR
os.environ["TMP"] = MP_TMPDIR

import re
import sys
import time
import json
import glob
import random
import logging
import tempfile
import traceback
import requests
import warnings
import numpy as np
import scipy.ndimage as measure
import torch

try:
    # 避免 DataLoader worker 通过 file_descriptor/DupFd 共享 tensor 时触发 AF_UNIX path 相关问题
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

from torch.utils.data import DataLoader
from tqdm import tqdm

import monai
from monai.data import PersistentDataset, list_data_collate, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    CropForegroundd,
    SpatialPadd,
    RandFlipd,
    RandGaussianNoised,
    AsDiscreted,
    EnsureTyped,
    MapTransform,
    KeepLargestConnectedComponentd,
    SelectItemsd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
)
from monai.networks.nets import DynUNet
from monai.losses import DiceFocalLoss, FocalLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warnings.filterwarnings("ignore")

# ==========================================
# 1. 基础运行配置
# ==========================================
RUN_DIR = os.path.expanduser("~/autodl-tmp/lits_stage2_v33_conservative")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_PATH = os.path.join(RUN_DIR, "train_stage2.log")
logger = logging.getLogger("LiTS-Stage2-v3.3-recall-calib")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.__stdout__)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

# 可复现性与速度二选一：
# True  更可复现，但会牺牲速度；False 速度更好。
DETERMINISTIC = False
if DETERMINISTIC:
    set_determinism(seed=42)
else:
    torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"


# ==========================================
# 2. 显存自适应训练配置
# ==========================================
def get_gpu_profile():
    if not torch.cuda.is_available():
        return {
            "name": "cpu",
            "PATCH_SIZE": (96, 96, 96),
            "BATCH_SIZE": 1,
            "NUM_SAMPLES": 1,
            "ACCUMULATION_STEPS": 1,
            "SW_BATCH_SIZE": 1,
        }

    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # 96GB 显存：优先增大 patch 上下文，而不是盲目增 batch
    if mem_gb >= 80:
        return {
            "name": f"{mem_gb:.0f}GB",
            "PATCH_SIZE": (160, 160, 128),
            "BATCH_SIZE": 1,
            "NUM_SAMPLES": 2,
            "ACCUMULATION_STEPS": 4,
            "SW_BATCH_SIZE": 4,
        }

    # 48GB 显存：稳定优先，真实 micro-batch = BATCH_SIZE * NUM_SAMPLES = 2
    if mem_gb >= 45:
        return {
            "name": f"{mem_gb:.0f}GB",
            "PATCH_SIZE": (128, 128, 128),
            "BATCH_SIZE": 1,
            "NUM_SAMPLES": 2,
            "ACCUMULATION_STEPS": 4,
            "SW_BATCH_SIZE": 2,
        }

    return {
        "name": f"{mem_gb:.0f}GB",
        "PATCH_SIZE": (96, 96, 96),
        "BATCH_SIZE": 1,
        "NUM_SAMPLES": 2,
        "ACCUMULATION_STEPS": 4,
        "SW_BATCH_SIZE": 1,
    }


GPU_PROFILE = get_gpu_profile()
PATCH_SIZE = GPU_PROFILE["PATCH_SIZE"]
BATCH_SIZE = GPU_PROFILE["BATCH_SIZE"]
NUM_SAMPLES = GPU_PROFILE["NUM_SAMPLES"]
ACCUMULATION_STEPS = GPU_PROFILE["ACCUMULATION_STEPS"]
SW_BATCH_SIZE = GPU_PROFILE["SW_BATCH_SIZE"]

PIXDIM = (1.0, 1.0, 1.5)

MAX_EPOCHS = 40
VAL_INTERVAL = 1
BASE_LR = 3e-6
WARMUP_EPOCHS = 2
PATIENCE = 8

# 由 eval_stage2_sweep_bias_minvol.py 得到的最佳验证/推理校准参数
# 当前 best_s2 sweep: Dice≈0.6790 at bias=0.75, min_tumor_vol=400
TUMOR_LOGIT_BIAS = 0.75
VAL_MIN_TUMOR_VOL = 400

ROOT_DIR = os.path.expanduser("~/autodl-tmp/LiTs")
STAGE1_SDF_DIR = os.path.expanduser("~/autodl-tmp/lits_stage1_sdf")
STAGE1_MODEL_PATH = os.path.expanduser("~/autodl-tmp/lits_heavy_train_192/best_heavy_model.pth")
# v3.3 conservative fine-tune: 从 v3.1 最优权重继续；先做 E0 基线验证，只在超过基线时保存。
PRETRAINED_STAGE2_PATH = os.path.expanduser("~/autodl-tmp/lits_stage2_optimized/best_s2.pth")

# v3.3 只做极轻微微调，pre-cache transform 与 v3.1 保持一致。
# 因此优先复用 v3.1 已经构建好的 PersistentDataset cache，避免重新写缓存把磁盘占满。
CACHE_BASENAME = f"persistent_cache_v4_patch{PATCH_SIZE[0]}x{PATCH_SIZE[1]}x{PATCH_SIZE[2]}_pix{PIXDIM[0]}_{PIXDIM[1]}_{PIXDIM[2]}"
DEFAULT_SHARED_CACHE_DIR = os.path.expanduser(
    os.path.join("~/autodl-tmp/lits_stage2_optimized", CACHE_BASENAME)
)
CACHE_DIR = os.environ.get("LITS_CACHE_DIR", DEFAULT_SHARED_CACHE_DIR)
if not os.path.exists(CACHE_DIR):
    # 兜底：如果旧 cache 不存在，才在当前 RUN_DIR 下新建。
    CACHE_DIR = os.path.join(RUN_DIR, CACHE_BASENAME)
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"🧊 Persistent cache dir: {CACHE_DIR}")
# 不要把 TMPDIR 指向 CACHE_DIR：路径太长会导致 multiprocessing 的 AF_UNIX socket 报错。
# PersistentDataset 的缓存目录仍然使用 CACHE_DIR；临时 socket 使用短路径 MP_TMPDIR。
tempfile.tempdir = MP_TMPDIR

# 使用短文件名，避免 token/长路径误伤 torch.save。
# 注意：不要把 WECHAT_TOKEN 拼进任何文件名。
CKPT_PATH = os.path.join(RUN_DIR, "ckpt_s2.pth")
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_s2.pth")
BEST_META_PATH = os.path.join(RUN_DIR, "best_s2_meta.json")

# 建议在命令行里设置：export AUTODL_WECHAT_TOKEN='你的token'
# 不建议把 token 明文写进脚本，更不要做全局替换。
WECHAT_TOKEN = os.environ.get("AUTODL_WECHAT_TOKEN", "")
AUTO_SHUTDOWN = True

for _p in [CKPT_PATH, BEST_MODEL_PATH, BEST_META_PATH]:
    if len(os.path.basename(_p)) > 100:
        raise RuntimeError(f"保存文件名异常过长，请检查是否误把 token 拼进路径: {_p}")


# ==========================================
# 3. 日志与通知机制
# ==========================================
def send_wechat(msg: str):
    if not WECHAT_TOKEN or WECHAT_TOKEN == "token":
        return
    try:
        clean_msg = str(msg).replace("\n", " ").replace("\r", " ")
        clean_msg = re.sub(r"[^\w\s\.\-\:\u4e00-\u9fa5]", "", clean_msg)
        if len(clean_msg) > 20:
            clean_msg = clean_msg[:17] + "..."
        requests.post(
            "https://www.autodl.com/api/v1/wechat/message/send",
            json={"title": "Cascade S2 v4", "name": clean_msg, "content": "Update"},
            headers={"Authorization": WECHAT_TOKEN},
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"WeChat Send Failed: {e}")


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_save(obj, path):
    """先写入短临时文件，再原子替换，避免 checkpoint 写一半损坏。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = os.path.basename(path)
    if len(base) > 100:
        raise RuntimeError(f"保存文件名异常过长，疑似 token 被拼进路径: {path}")
    tmp_path = os.path.join(os.path.dirname(path), f".{base}.tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


# ==========================================
# 4. 自定义数据算子
# ==========================================
class EnsureSDFChannelFirstd(MapTransform):
    """确保 SDF 为 channel-first: C, H, W, D。"""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]

            if isinstance(x, torch.Tensor):
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                elif x.ndim == 4:
                    if x.shape[0] in (1, 2, 3, 4):
                        pass
                    elif x.shape[-1] in (1, 2, 3, 4):
                        x = torch.movedim(x, -1, 0).contiguous()
                    else:
                        raise RuntimeError(f"SDF shape 无法判断通道维: {tuple(x.shape)}")
                else:
                    raise RuntimeError(f"SDF ndim 异常: {x.ndim}, shape={tuple(x.shape)}")
            else:
                x = np.asarray(x)
                if x.ndim == 3:
                    x = np.expand_dims(x, axis=0)
                elif x.ndim == 4:
                    if x.shape[0] in (1, 2, 3, 4):
                        pass
                    elif x.shape[-1] in (1, 2, 3, 4):
                        x = np.moveaxis(x, -1, 0)
                    else:
                        raise RuntimeError(f"SDF shape 无法判断通道维: {x.shape}")
                else:
                    raise RuntimeError(f"SDF ndim 异常: {x.ndim}, shape={x.shape}")

            d[key] = x
        return d


class TruncateSDFNormd(MapTransform):
    def __init__(self, keys, truncation=20.0):
        super().__init__(keys)
        self.trunc = float(truncation)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            sdf = d[key]
            if not isinstance(sdf, torch.Tensor):
                sdf = torch.as_tensor(sdf)
            sdf = torch.clamp(sdf.float(), min=-self.trunc, max=self.trunc) / self.trunc
            d[key] = sdf
        return d


class CastLabelToLongd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])
            d[key] = d[key].long()
        return d


class RandSDFDropoutd(MapTransform):
    def __init__(self, keys, prob=0.2):
        super().__init__(keys)
        self.prob = float(prob)

    def _process(self, d):
        d = dict(d)
        if random.random() < self.prob:
            for key in self.keys:
                d[key] = torch.zeros_like(d[key])
        return d

    def __call__(self, data):
        if isinstance(data, list):
            return [self._process(i) for i in data]
        return self._process(data)


class StrictWindowNormd(MapTransform):
    """
    CT window + foreground z-score。

    注意：训练随机窗宽窗位不能放进 PersistentDataset 的 pre_trans，
    否则第一次 cache 后就固定了。
    """

    def __init__(self, keys, train=True):
        super().__init__(keys)
        self.train = bool(train)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            img = img.float()

            if self.train and random.random() > 0.5:
                # 回到 v2.2 的肝脏/肿瘤 CT window，避免 v3 窄窗造成分布偏移
                w_center = random.uniform(100, 150)
                w_width = random.uniform(300, 500)
            else:
                w_center, w_width = 150.0, 500.0

            min_hu = w_center - w_width / 2.0
            max_hu = w_center + w_width / 2.0

            img = torch.clamp(img, min_hu, max_hu)
            fg = img[img > min_hu]

            if fg.numel() > 1:
                img = (img - fg.mean()) / (fg.std(unbiased=False) + 1e-8)
            else:
                img = torch.zeros_like(img)

            d[key] = img.float()
        return d


class AddTumorLogitBiasd(MapTransform):
    """在 argmax 前给 tumor 通道加 logit bias，用 sweep 得到的校准值提升 recall。"""

    def __init__(self, keys, bias=0.0, tumor_channel=2):
        super().__init__(keys)
        self.bias = float(bias)
        self.tumor_channel = int(tumor_channel)

    def __call__(self, data):
        d = dict(data)
        if abs(self.bias) < 1e-12:
            return d
        for key in self.keys:
            x = d[key]
            if isinstance(x, torch.Tensor):
                # decollate 后通常是 C,H,W,D；保险兼容 B,C,H,W,D
                y = x.clone()
                if y.ndim >= 4 and y.shape[0] > self.tumor_channel:
                    y[self.tumor_channel] = y[self.tumor_channel] + self.bias
                elif y.ndim >= 5 and y.shape[1] > self.tumor_channel:
                    y[:, self.tumor_channel] = y[:, self.tumor_channel] + self.bias
                d[key] = y
        return d


class CCAPostProcessingd(MapTransform):
    """v2.2 后处理：最大肝脏连通域 + 肿瘤限制在肝脏内 + 删除极小肿瘤小岛。"""

    def __init__(self, keys, min_tumor_vol=200):
        super().__init__(keys)
        self.min_tumor_vol = int(min_tumor_vol)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            pred_tensor = d[key]
            device = pred_tensor.device if isinstance(pred_tensor, torch.Tensor) else None

            # 支持 one-hot C,H,W,D 或 label map H,W,D / 1,H,W,D
            if isinstance(pred_tensor, torch.Tensor):
                x = pred_tensor.detach()
                if x.ndim == 4 and x.shape[0] == 3:
                    pred_np = torch.argmax(x, dim=0).cpu().numpy()
                    return_onehot = True
                elif x.ndim == 4 and x.shape[0] == 1:
                    pred_np = x.squeeze(0).cpu().numpy()
                    return_onehot = False
                else:
                    pred_np = x.cpu().numpy()
                    return_onehot = False
            else:
                pred_np = np.asarray(pred_tensor)
                return_onehot = False

            liver_mask = (pred_np == 1) | (pred_np == 2)
            labels_liver, num_liver = measure.label(liver_mask)
            if num_liver > 0:
                largest_liver_label = 1 + np.argmax(np.bincount(labels_liver.flat)[1:])
                valid_liver = labels_liver == largest_liver_label
            else:
                valid_liver = liver_mask

            tumor_mask = (pred_np == 2) & valid_liver
            labels_tumor, num_tumor = measure.label(tumor_mask)
            for i in range(1, num_tumor + 1):
                if np.sum(labels_tumor == i) < self.min_tumor_vol:
                    tumor_mask[labels_tumor == i] = False

            final_pred = np.zeros(pred_np.shape, dtype=np.int64)
            final_pred[valid_liver] = 1
            final_pred[tumor_mask] = 2
            final_tensor = torch.from_numpy(final_pred).to(device if device is not None else "cpu")

            if return_onehot:
                final_tensor = torch.nn.functional.one_hot(final_tensor.long(), num_classes=3)
                final_tensor = final_tensor.permute(3, 0, 1, 2).float()
            else:
                # MONAI AsDiscreted(to_onehot=...) 要求 label map 带单通道维: 1,H,W,D
                # argmax 后通常是 H,W,D；这里补回通道维，避免验证时报
                # AssertionError: labels should have a channel with length equal to one.
                if final_tensor.ndim == 3:
                    final_tensor = final_tensor.unsqueeze(0)

            d[key] = final_tensor
        return d


class Stage2Loss(torch.nn.Module):
    """回到 v2.2 的 GeneralizedDice + Focal。"""

    def __init__(self):
        super().__init__()
        self.gdice = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
        self.focal = FocalLoss(gamma=2.5, to_onehot_y=True, use_softmax=True)

    def forward(self, pred, target):
        return 0.6 * self.gdice(pred, target) + 0.4 * self.focal(pred, target)


class AugWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, aug_transform):
        self.dataset = dataset
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.aug_transform(self.dataset[i])


# ==========================================
# 5. 数据管线
# ==========================================
def get_transforms():
    pre_trans = Compose([
        LoadImaged(keys=["image", "label", "sdf"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureSDFChannelFirstd(keys=["sdf"]),
        TruncateSDFNormd(keys=["sdf"], truncation=20.0),
        Orientationd(keys=["image", "label", "sdf"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label", "sdf"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest", "bilinear"),
        ),
        CropForegroundd(
            keys=["image", "label", "sdf"],
            source_key="image",
            select_fn=lambda x: x > -500,
            margin=15,
        ),
        SpatialPadd(keys=["image", "label", "sdf"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image", "label", "sdf"], track_meta=False),
        CastLabelToLongd(keys=["label"]),
        SelectItemsd(keys=["image", "label", "sdf"]),
    ])

    train_aug = Compose([
        # 先恢复 v2.2 的温和阳性采样，避免 v3 ratios=[1,2,6] 造成肿瘤假阳性爆炸
        RandCropByPosNegLabeld(
            keys=["image", "label", "sdf"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=2,
            neg=1,
            num_samples=NUM_SAMPLES,
        ),
        StrictWindowNormd(keys=["image"], train=True),
        RandFlipd(keys=["image", "label", "sdf"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label", "sdf"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label", "sdf"], prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys=["image"], prob=0.15, std=0.1),
        RandSDFDropoutd(keys=["sdf"], prob=0.05),
        EnsureTyped(keys=["image", "label", "sdf"], track_meta=False),
        CastLabelToLongd(keys=["label"]),
        SelectItemsd(keys=["image", "label", "sdf"]),
    ])

    val_pre_trans = Compose([
        LoadImaged(keys=["image", "label", "sdf"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureSDFChannelFirstd(keys=["sdf"]),
        TruncateSDFNormd(keys=["sdf"], truncation=20.0),
        Orientationd(keys=["image", "label", "sdf"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label", "sdf"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest", "bilinear"),
        ),
        CropForegroundd(
            keys=["image", "label", "sdf"],
            source_key="image",
            select_fn=lambda x: x > -500,
            margin=15,
        ),
        StrictWindowNormd(keys=["image"], train=False),
        EnsureTyped(keys=["image", "label", "sdf"], track_meta=False),
        CastLabelToLongd(keys=["label"]),
        SelectItemsd(keys=["image", "label", "sdf"]),
    ])

    return pre_trans, train_aug, val_pre_trans


def build_dataloaders(train_files, val_files):
    pre_trans, train_aug, val_pre_trans = get_transforms()

    train_ds_base = PersistentDataset(
        data=train_files,
        transform=pre_trans,
        cache_dir=os.path.join(CACHE_DIR, "train"),
    )
    val_ds = PersistentDataset(
        data=val_files,
        transform=val_pre_trans,
        cache_dir=os.path.join(CACHE_DIR, "val"),
    )

    train_workers = int(os.environ.get("LITS_NUM_WORKERS", "4"))
    val_workers = max(1, train_workers // 2) if train_workers > 0 else 0

    train_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=AMP_ENABLED,
        collate_fn=list_data_collate,
        persistent_workers=train_workers > 0,
    )
    if train_workers > 0:
        train_kwargs["prefetch_factor"] = 2

    val_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=AMP_ENABLED,
        persistent_workers=val_workers > 0,
    )
    if val_workers > 0:
        val_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(AugWrapper(train_ds_base, train_aug), **train_kwargs)
    val_loader = DataLoader(val_ds, **val_kwargs)
    return train_loader, val_loader


# ==========================================
# 6. 模型与权重迁移
# ==========================================
def extract_main_output(y):
    if torch.is_tensor(y) and y.ndim == 6:
        return torch.unbind(y, dim=1)[0]
    if isinstance(y, (list, tuple)):
        return y[0]
    return y


def strip_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "network", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    out = {}
    for k, v in ckpt.items():
        out[k.replace("module.", "")] = v
    return out


def get_model(in_channels):
    # 与 v2.2 / Stage1 保持同构：动态生成 6 层左右 DynUNet，而不是 v3 写死 5 层。
    strides, kernels = [[1, 1, 1]], [[3, 3, 3]]
    curr_size = list(PATCH_SIZE)
    while all(x > 4 for x in curr_size) and len(strides) < 6:
        st = [2 if x >= 8 else 1 for x in curr_size]
        strides.append(st)
        kernels.append([3, 3, 3])
        curr_size = [x // y for x, y in zip(curr_size, st)]

    model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=3,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=[32 * (2 ** i) for i in range(len(strides))],
        dropout=0.1,
        norm_name="instance",
        res_block=True,
        deep_supervision=True,
    ).to(DEVICE)

    if os.path.exists(STAGE1_MODEL_PATH):
        logger.info("🔄 正在执行 Stage1 -> Stage2 权重迁移...")
        old_state = strip_state_dict(safe_torch_load(STAGE1_MODEL_PATH, map_location=DEVICE))
        new_state = model.state_dict()

        inherited_layers = 0
        bridged_layers = 0

        for k, v in old_state.items():
            if k not in new_state:
                continue

            if v.shape == new_state[k].shape:
                new_state[k] = v
                inherited_layers += 1
                continue

            # 桥接所有输入卷积：Stage1 单通道 CT -> Stage2 CT+SDF 多通道。
            # 不再只桥接第一个 conv，否则会少继承多个 input/skip 分支。
            if (
                v.ndim == 5
                and new_state[k].ndim == 5
                and v.shape[1] == 1
                and new_state[k].shape[1] == in_channels
                and v.shape[0] == new_state[k].shape[0]
                and v.shape[2:] == new_state[k].shape[2:]
            ):
                new_w = new_state[k].clone()
                new_w[:, 0:1] = v
                if in_channels > 1:
                    # SDF 通道初始置零，先保持 Stage1 CT 行为，再让微调慢慢学习 SDF。
                    new_w[:, 1:] = 0.0
                new_state[k] = new_w
                inherited_layers += 1
                bridged_layers += 1
                logger.info(f"   => 桥接输入层 {k}: {tuple(v.shape)} -> {tuple(new_w.shape)}")

        model.load_state_dict(new_state, strict=True)
        logger.info(f"✅ 完成迁移学习：完全/部分继承 {inherited_layers} 层，其中输入桥接 {bridged_layers} 层")

    return model

class DeepSupervisionLoss(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, preds, target):
        if torch.is_tensor(preds) and preds.ndim == 6:
            preds = torch.unbind(preds, dim=1)

        if not isinstance(preds, (list, tuple)):
            return self.criterion(preds, target)

        weights = [1.0 / (2 ** i) for i in range(len(preds))]
        loss = 0.0

        for i, pred in enumerate(preds):
            current_target = target
            if pred.shape[2:] != target.shape[2:]:
                current_target = torch.nn.functional.interpolate(
                    target.float(),
                    size=pred.shape[2:],
                    mode="nearest",
                ).long()

            loss = loss + weights[i] * self.criterion(pred, current_target)

        return loss / sum(weights)


def try_resume_checkpoint(model, optimizer, scheduler, scaler):
    start_epoch, best_metric, patience_counter = 0, -1.0, 0

    if not os.path.exists(CKPT_PATH):
        return start_epoch, best_metric, patience_counter

    try:
        ckpt = safe_torch_load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sch"])
        if AMP_ENABLED and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best", -1.0))
        patience_counter = int(ckpt.get("patience", 0))
        logger.info(f"🔄 检测到训练中断，从 Epoch {start_epoch} 恢复，best={best_metric:.4f}")
    except Exception as e:
        logger.warning(f"⚠️ 旧 checkpoint 与当前配置不兼容，跳过恢复: {e}")
        logger.warning("   如确认要重训，可手动删除 checkpoint_stage2.pth")

    return start_epoch, best_metric, patience_counter


# ==========================================
# 7. 数据索引
# ==========================================
def extract_case_id(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    if not nums:
        raise RuntimeError(f"无法从文件名提取 case id: {path}")
    return int(nums[0])


def build_file_list():
    all_imgs = sorted(
        glob.glob(os.path.join(ROOT_DIR, "**", "volume-*.nii*"), recursive=True),
        key=extract_case_id,
    )
    all_labs = sorted(
        glob.glob(os.path.join(ROOT_DIR, "segmentations", "segmentation-*.nii*")),
        key=extract_case_id,
    )
    all_sdf = sorted(
        glob.glob(os.path.join(STAGE1_SDF_DIR, "*_sdf.nii*")),
        key=extract_case_id,
    )

    img_map = {extract_case_id(p): p for p in all_imgs}
    lab_map = {extract_case_id(p): p for p in all_labs}
    sdf_map = {extract_case_id(p): p for p in all_sdf}

    valid_ids = sorted(set(img_map) & set(lab_map) & set(sdf_map))
    missing_lab = sorted(set(img_map) - set(lab_map))
    missing_sdf = sorted(set(img_map) - set(sdf_map))

    if missing_lab:
        logger.warning(f"有 {len(missing_lab)} 个 image 缺少 label，示例: {missing_lab[:5]}")
    if missing_sdf:
        logger.warning(f"有 {len(missing_sdf)} 个 image 缺少 sdf，示例: {missing_sdf[:5]}")

    data = [{"image": img_map[i], "label": lab_map[i], "sdf": sdf_map[i], "case_id": i} for i in valid_ids]
    random.seed(42)
    random.shuffle(data)
    return data


# ==========================================
# 8. 主训练与验证循环
# ==========================================
def main():
    logger.info(f"🚀 Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    data = build_file_list()
    if len(data) < 2:
        logger.warning("未检测到足够有效数据，请检查 ROOT_DIR 和 STAGE1_SDF_DIR 路径是否正确！")
        return

    val_n = min(20, max(1, int(len(data) * 0.15)))
    train_files, val_files = data[:-val_n], data[-val_n:]

    if len(train_files) == 0:
        logger.warning("训练集为空，请检查数据数量。")
        return

    logger.info(f"📊 成功挂载级联数据: 训练集 {len(train_files)} / 验证集 {len(val_files)}")
    logger.info(
        f"🧪 GPU Profile: {GPU_PROFILE['name']} | PATCH={PATCH_SIZE} | "
        f"BATCH={BATCH_SIZE} | NUM_SAMPLES={NUM_SAMPLES} | ACC={ACCUMULATION_STEPS} | "
        f"micro_batch={BATCH_SIZE * NUM_SAMPLES} | "
        f"effective_batch={BATCH_SIZE * NUM_SAMPLES * ACCUMULATION_STEPS} | "
        f"SW_BATCH={SW_BATCH_SIZE}"
    )

    train_loader, val_loader = build_dataloaders(train_files, val_files)

    # 取一个 batch 做通道检查，避免 SDF 通道数与模型输入不一致
    probe_batch = next(iter(train_loader))
    img_ch = int(probe_batch["image"].shape[1])
    sdf_ch = int(probe_batch["sdf"].shape[1])
    in_channels = img_ch + sdf_ch

    logger.info(f"🧪 Channel check: image={img_ch}, sdf={sdf_ch}, model_in={in_channels}")
    logger.info(
        f"🧪 Probe shapes: image={tuple(probe_batch['image'].shape)}, "
        f"sdf={tuple(probe_batch['sdf'].shape)}, label={tuple(probe_batch['label'].shape)}"
    )

    if img_ch != 1:
        raise RuntimeError(f"image 通道异常，期望 1，实际 {img_ch}")
    if in_channels < 2:
        raise RuntimeError(f"输入通道异常: image={img_ch}, sdf={sdf_ch}")

    del probe_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = get_model(in_channels=in_channels)

    # v3.3: 从 v3.1 的 best_s2.pth 继续做极保守 fine-tune。
    # 新 RUN_DIR 不会误读旧 ckpt，但会显式加载旧 best 权重；E0 先验证并保存基线。
    if os.path.exists(PRETRAINED_STAGE2_PATH):
        logger.info(f"🔄 从 v3.1 最优模型继续微调: {PRETRAINED_STAGE2_PATH}")
        stage2_state = strip_state_dict(safe_torch_load(PRETRAINED_STAGE2_PATH, map_location=DEVICE))
        incompat = model.load_state_dict(stage2_state, strict=False)
        logger.info(
            f"✅ 已加载 v3.1 best；missing={len(incompat.missing_keys)}, "
            f"unexpected={len(incompat.unexpected_keys)}"
        )
    else:
        logger.warning(f"未找到 v3.1 best 权重，将仅使用 Stage1 迁移: {PRETRAINED_STAGE2_PATH}")

    # 回到 v2.2 的 loss；v3 的 DiceFocalLoss(include_background=False) 对早期假阳性更敏感.
    loss_function = DeepSupervisionLoss(Stage2Loss())

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, MAX_EPOCHS - WARMUP_EPOCHS))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    dice_metric_tumor = DiceMetric(include_background=True, reduction="mean")

    logger.info(f"🎯 Validation calibration: tumor_logit_bias={TUMOR_LOGIT_BIAS}, min_tumor_vol={VAL_MIN_TUMOR_VOL}")

    post_pred = Compose([
        EnsureTyped(keys=["pred"]),
        # 关键：bias 必须在 argmax/CCA 前加到 logits 的 tumor 通道上。
        AddTumorLogitBiasd(keys=["pred"], bias=TUMOR_LOGIT_BIAS, tumor_channel=2),
        # 输入仍是 C,H,W,D logits；CCA 内部 argmax，并返回 one-hot C,H,W,D。
        CCAPostProcessingd(keys=["pred"], min_tumor_vol=VAL_MIN_TUMOR_VOL),
    ])

    start_epoch, best_metric, patience_counter = try_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    def run_validation(tag: str):
        model.eval()
        val_pred_tumor_voxels = 0.0
        val_label_tumor_voxels = 0.0
        val_cases = 0
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc=f"Validation {tag}", leave=False):
                v_images = val_data["image"].to(DEVICE, non_blocking=True)
                v_sdfs = val_data["sdf"].to(DEVICE, non_blocking=True)
                v_labels = val_data["label"].to(DEVICE, non_blocking=True)

                v_inputs = torch.cat([v_images, v_sdfs], dim=1)

                def val_predictor(x):
                    return extract_main_output(model(x))

                with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                    v_outputs = sliding_window_inference(
                        inputs=v_inputs,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=SW_BATCH_SIZE,
                        predictor=val_predictor,
                        overlap=0.5,
                        mode="gaussian",
                    )

                val_data["pred"] = v_outputs
                val_data_list = decollate_batch(val_data)
                val_data_list = [post_pred(i) for i in val_data_list]

                processed_pred = val_data_list[0]["pred"]
                tumor_pred = processed_pred[2:3, ...].unsqueeze(0)
                tumor_label = (v_labels == 2).float()

                val_pred_tumor_voxels += float(tumor_pred.sum().item())
                val_label_tumor_voxels += float(tumor_label.sum().item())
                val_cases += 1

                dice_metric_tumor(y_pred=tumor_pred, y=tumor_label)

            metric_tumor = float(dice_metric_tumor.aggregate().item())
            dice_metric_tumor.reset()

        return metric_tumor, val_pred_tumor_voxels, val_label_tumor_voxels, val_cases

    # E0 先验证 v3.1 best + 校准参数，作为安全基线。后续微调只有超过它才会保存。
    if start_epoch == 0 and best_metric < 0:
        metric_tumor, pred_vox, label_vox, val_cases = run_validation("E0")
        best_metric = metric_tumor
        patience_counter = 0
        logger.info(
            f"E0 | Tumor Dice: {metric_tumor:.4f} | "
            f"PredTumorVox: {pred_vox:.0f} | "
            f"LabelTumorVox: {label_vox:.0f} | "
            f"ValCases: {val_cases} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        safe_save(model.state_dict(), BEST_MODEL_PATH)
        with open(BEST_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"epoch": 0, "tumor_dice": best_metric, "note": "v3.1 best baseline with v3.3 calibration"}, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 已保存 E0 安全基线：Tumor Dice={best_metric:.4f}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}", leave=False)
        for step, batch_data in enumerate(pbar):
            images = batch_data["image"].to(DEVICE, non_blocking=True)
            sdfs = batch_data["sdf"].to(DEVICE, non_blocking=True)
            labels = batch_data["label"].to(DEVICE, non_blocking=True)

            inputs = torch.cat([images, sdfs], dim=1)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step_loss = float(loss.item() * ACCUMULATION_STEPS)
            epoch_loss += step_loss
            pbar.set_postfix({"Loss": f"{step_loss:.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        logger.info(f"E{epoch + 1} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_pred_tumor_voxels = 0.0
            val_label_tumor_voxels = 0.0
            val_cases = 0
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation", leave=False):
                    v_images = val_data["image"].to(DEVICE, non_blocking=True)
                    v_sdfs = val_data["sdf"].to(DEVICE, non_blocking=True)
                    v_labels = val_data["label"].to(DEVICE, non_blocking=True)

                    v_inputs = torch.cat([v_images, v_sdfs], dim=1)

                    def val_predictor(x):
                        return extract_main_output(model(x))

                    with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                        v_outputs = sliding_window_inference(
                            inputs=v_inputs,
                            roi_size=PATCH_SIZE,
                            sw_batch_size=SW_BATCH_SIZE,
                            predictor=val_predictor,
                            overlap=0.5,
                            mode="gaussian",
                        )

                    val_data["pred"] = v_outputs
                    val_data_list = decollate_batch(val_data)
                    val_data_list = [post_pred(i) for i in val_data_list]

                    processed_pred = val_data_list[0]["pred"]
                    tumor_pred = processed_pred[2:3, ...].unsqueeze(0)
                    tumor_label = (v_labels == 2).float()

                    val_pred_tumor_voxels += float(tumor_pred.sum().item())
                    val_label_tumor_voxels += float(tumor_label.sum().item())
                    val_cases += 1

                    dice_metric_tumor(y_pred=tumor_pred, y=tumor_label)

                metric_tumor = float(dice_metric_tumor.aggregate().item())
                dice_metric_tumor.reset()

            logger.info(
                f"E{epoch + 1} | Tumor Dice: {metric_tumor:.4f} | "
                f"PredTumorVox: {val_pred_tumor_voxels:.0f} | "
                f"LabelTumorVox: {val_label_tumor_voxels:.0f} | "
                f"ValCases: {val_cases} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            if metric_tumor > best_metric:
                send_wechat(f"S2_v33 E{epoch + 1} T_Dice:{metric_tumor:.3f}")
                best_metric = metric_tumor
                patience_counter = 0
                safe_save(model.state_dict(), BEST_MODEL_PATH)
                with open(BEST_META_PATH, "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch + 1, "tumor_dice": best_metric}, f, ensure_ascii=False, indent=2)
                logger.info(f"★★★ 破纪录！当前最高 Tumor Dice: {best_metric:.4f} ★★★")
            else:
                patience_counter += 1
                logger.info(f"早停计数: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    logger.info("验证集指标不再提升，触发早停。")
                    break

        safe_save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "sch": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best_metric,
                "patience": patience_counter,
                "gpu_profile": GPU_PROFILE,
                "patch_size": PATCH_SIZE,
                "pixdim": PIXDIM,
            },
            CKPT_PATH,
        )

    logger.info(f"训练结束。Best Tumor Dice: {best_metric:.4f}")


if __name__ == "__main__":
    try:
        main()
        send_wechat("Train V4 Success!")
    except Exception as e:
        error_info = str(e).split(":")[-1].strip()
        send_wechat(f"Err:{error_info[:15]}")
        logger.error(traceback.format_exc())
    finally:
        if AUTO_SHUTDOWN:
            logger.info("系统将在 10 秒后自动关机...")
            time.sleep(10)
            os.system("/usr/bin/shutdown")
