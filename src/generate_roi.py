import os
import re
import json
from glob import glob

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
    SpatialCropd,
)
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from monai.transforms.utils import generate_spatial_bounding_box


# =========================
# 配置
# =========================
STAGE1_MODEL_PATH = os.path.expanduser(
    "~/autodl-tmp/two_stage_liver_model/stage1_liver_roi_model.pth"
)
RAW_DATA_DIR = os.path.expanduser("~/autodl-tmp/LiTs")
ROI_OUT_DIR = os.path.expanduser("~/autodl-tmp/LiTs_ROI")

PATCH_SIZE = (192, 192, 192)
PIXDIM = (1.0, 1.0, 1.5)
FEAT_SIZE = 32
NUM_CLASSES = 2  # channel 0: liver, channel 1: tumor
BBOX_MARGIN = 24
SW_OVERLAP = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_OUT_DIR = os.path.join(ROI_OUT_DIR, "images")
LAB_OUT_DIR = os.path.join(ROI_OUT_DIR, "labels")
BBOX_OUT_DIR = os.path.join(ROI_OUT_DIR, "bbox_json")

os.makedirs(ROI_OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(LAB_OUT_DIR, exist_ok=True)
os.makedirs(BBOX_OUT_DIR, exist_ok=True)


# =========================
# 工具函数
# =========================
def extract_number(path: str) -> int:
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[0]) if nums else -1


def build_case_list(raw_data_dir: str):
    img_files = sorted(
        glob(os.path.join(raw_data_dir, "**", "volume-*.nii*"), recursive=True),
        key=extract_number,
    )
    lab_files = sorted(
        glob(os.path.join(raw_data_dir, "segmentations", "segmentation-*.nii*")),
        key=extract_number,
    )

    if len(img_files) == 0:
        raise RuntimeError(f"未找到图像文件: {raw_data_dir}")
    if len(lab_files) == 0:
        raise RuntimeError(f"未找到标签文件: {os.path.join(raw_data_dir, 'segmentations')}")

    img_map = {extract_number(p): p for p in img_files}
    lab_map = {extract_number(p): p for p in lab_files}

    valid_ids = sorted(set(img_map.keys()) & set(lab_map.keys()))
    if len(valid_ids) == 0:
        raise RuntimeError("没有找到可配对的 image / label")

    cases = []
    for cid in valid_ids:
        cases.append(
            {
                "id": cid,
                "image": img_map[cid],
                "label": lab_map[cid],
            }
        )
    return cases


def build_stage1_model():
    strides, kernels = [[1, 1, 1]], [[3, 3, 3]]
    curr_size = list(PATCH_SIZE)
    while all(s > 4 for s in curr_size) and len(strides) < 6:
        s = [2 if i >= 8 else 1 for i in curr_size]
        strides.append(s)
        kernels.append([3, 3, 3])
        curr_size = [i // j for i, j in zip(curr_size, s)]

    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=[FEAT_SIZE * (2 ** i) for i in range(len(strides))],
        dropout=0.1,
        norm_name="instance",
        deep_supervision=True,
        res_block=True,
    ).to(DEVICE)

    ckpt = torch.load(os.path.expanduser(STAGE1_MODEL_PATH), map_location=DEVICE)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def predictor_wrapper(model, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    if torch.is_tensor(y) and y.ndim == 6:
        return y[:, 0]
    if isinstance(y, (list, tuple)):
        return y[0]
    return y


def make_preprocess():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=PIXDIM,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def clamp_bbox(start, end, shape):
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    shape = np.asarray(shape, dtype=np.int64)

    start = np.maximum(start, 0)
    end = np.minimum(end, shape)

    # 防止空 bbox
    for i in range(len(shape)):
        if end[i] <= start[i]:
            end[i] = min(start[i] + 1, shape[i])
    return start.tolist(), end.tolist()


# =========================
# 主逻辑
# =========================
def main():
    print(f"DEVICE: {DEVICE}")
    print(f"STAGE1_MODEL_PATH: {STAGE1_MODEL_PATH}")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"ROI_OUT_DIR: {ROI_OUT_DIR}")

    if not os.path.exists(STAGE1_MODEL_PATH):
        raise FileNotFoundError(f"未找到 Stage1 模型: {STAGE1_MODEL_PATH}")

    cases = build_case_list(RAW_DATA_DIR)
    print(f"共找到 {len(cases)} 个可配对 case")

    model = build_stage1_model()
    process_trans = make_preprocess()

    with torch.no_grad():
        for case in tqdm(cases, desc="Extracting ROI"):
            cid = case["id"]
            img_path = case["image"]
            lab_path = case["label"]

            # A. 预处理，保证 image / label 空间完全对齐
            data = process_trans({"image": img_path, "label": lab_path})

            # B. Stage1 推理 whole liver
            input_tensor = data["image"].unsqueeze(0).to(DEVICE)  # [1,1,D,H,W]

            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=PATCH_SIZE,
                sw_batch_size=1,
                predictor=lambda x: predictor_wrapper(model, x),
                overlap=SW_OVERLAP,
            )

            # C. 取 liver 通道
            # 约定 channel 0 是 liver / whole liver
            liver_logits = output[0, 0]
            liver_mask = (torch.sigmoid(liver_logits) > 0.5).float()

            mask_np = liver_mask.detach().cpu().numpy()
            if not np.any(mask_np):
                print(f"[Skip] Case {cid}: 未预测出肝脏区域")
                continue

            # D. 根据 liver mask 算 bbox，并加 margin
            start, end = generate_spatial_bounding_box(mask_np[None], margin=BBOX_MARGIN)


            # 注意 generate_spatial_bounding_box 返回的是空间维度
            # 对当前 [D,H,W] mask，对应 roi_start / roi_end 的 3 个维度
            spatial_shape = np.array(mask_np.shape, dtype=np.int64)
            start, end = clamp_bbox(start, end, spatial_shape)

            # E. 对预处理后的 image / label 做同样裁剪
            cropper = SpatialCropd(
                keys=["image", "label"],
                roi_start=start,
                roi_end=end,
            )
            roi_data = cropper(data)

            # F. 保存 ROI
            img_name = f"volume-{cid}_roi.nii.gz"
            lab_name = f"segmentation-{cid}_roi.nii.gz"
            bbox_name = f"case-{cid}_bbox.json"
            
            roi_img_tensor = roi_data["image"]
            roi_lab_tensor = roi_data["label"]
            
            roi_img = roi_img_tensor.squeeze(0).detach().cpu().numpy()
            roi_lab = roi_lab_tensor.squeeze(0).detach().cpu().numpy()
            
            # 兼容新旧 MONAI 元数据格式
            affine = None
            if hasattr(roi_img_tensor, "meta") and roi_img_tensor.meta is not None:
                affine = roi_img_tensor.meta.get("affine", None)
            
            if affine is None and "image_meta_dict" in roi_data:
                affine = roi_data["image_meta_dict"].get("affine", None)
            
            if affine is None:
                affine = np.eye(4, dtype=np.float32)
            
            if isinstance(affine, torch.Tensor):
                affine = affine.detach().cpu().numpy()

            nib.save(
                nib.Nifti1Image(roi_img.astype(np.float32), affine),
                os.path.join(IMG_OUT_DIR, img_name),
            )
            nib.save(
                nib.Nifti1Image(roi_lab.astype(np.int16), affine),
                os.path.join(LAB_OUT_DIR, lab_name),
            )

            bbox_info = {
                "case_id": cid,
                "image_path": img_path,
                "label_path": lab_path,
                "roi_image_path": os.path.join(IMG_OUT_DIR, img_name),
                "roi_label_path": os.path.join(LAB_OUT_DIR, lab_name),
                "roi_start_dhw": start,
                "roi_end_dhw": end,
                "margin": BBOX_MARGIN,
                "pixdim": list(PIXDIM),
                "patch_size_stage1": list(PATCH_SIZE),
                "roi_shape_dhw": list(roi_img.shape),
            }

            with open(os.path.join(BBOX_OUT_DIR, bbox_name), "w", encoding="utf-8") as f:
                json.dump(bbox_info, f, ensure_ascii=False, indent=2)

            # 释放显存
            del data, input_tensor, output, liver_logits, liver_mask, roi_data
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nROI 提取完成，输出目录: {ROI_OUT_DIR}")
    print(f"图像目录: {IMG_OUT_DIR}")
    print(f"标签目录: {LAB_OUT_DIR}")
    print(f"BBox目录: {BBOX_OUT_DIR}")


if __name__ == "__main__":
    main()
