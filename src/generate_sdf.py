import os

# 防止 scipy / numpy 自己偷偷多线程吃爆内存
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import time
import requests
import numpy as np
import scipy.ndimage as ndi
import nibabel as nib

from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


WECHAT_TOKEN = "YOUR_TOKEN"
AUTO_SHUTDOWN = True


# ================= 配置区 =================
NUM_CLASSES = 3
SDF_CLIP = 50.0
NUM_WORKERS = 4

PREDS_DIR = os.path.expanduser("~/autodl-tmp/lits_stage1_preds")
OUTPUT_DIR = os.path.expanduser("~/autodl-tmp/lits_stage1_sdf")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================


def make_output_name(filepath):
    filename = os.path.basename(filepath)

    if filename.endswith(".nii.gz"):
        return filename.replace(".nii.gz", "_sdf.nii.gz")

    if filename.endswith(".nii"):
        return filename.replace(".nii", "_sdf.nii")

    return filename + "_sdf.nii.gz"


def compute_binary_sdf(mask_bool, clip=50.0):
    """
    内部为正，外部为负。
    输出范围 [-1, 1]。
    """
    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=np.float32)

    inside = ndi.distance_transform_edt(mask_bool)
    outside = ndi.distance_transform_edt(~mask_bool)

    sdf = inside - outside
    sdf = np.clip(sdf, -clip, clip) / clip

    return sdf.astype(np.float32)


def process_single_case(filepath):
    filename = os.path.basename(filepath)
    out_name = make_output_name(filepath)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if os.path.exists(out_path):
        return "skip"

    nii = nib.load(filepath)
    mask = np.asanyarray(nii.dataobj).astype(np.int16)

    unique_vals = np.unique(mask)
    if not np.all(np.isin(unique_vals, np.arange(NUM_CLASSES))):
        raise ValueError(f"{filename} 存在非法标签: {unique_vals}")

    h, w, d = mask.shape

    # 保存为 [H, W, D, 3]
    # channel 0: background，占位 0
    # channel 1: liver SDF
    # channel 2: tumor SDF
    sdf_arr = np.zeros((h, w, d, NUM_CLASSES), dtype=np.float32)

    liver_mask = mask == 1
    tumor_mask = mask == 2

    sdf_arr[..., 1] = compute_binary_sdf(liver_mask, SDF_CLIP)
    sdf_arr[..., 2] = compute_binary_sdf(tumor_mask, SDF_CLIP)

    new_nii = nib.Nifti1Image(sdf_arr, nii.affine)
    new_nii.header.set_data_dtype(np.float32)

    tmp_path = out_path + ".tmp.nii.gz"
    nib.save(new_nii, tmp_path)
    os.replace(tmp_path, out_path)

    del nii, mask, sdf_arr, liver_mask, tumor_mask
    gc.collect()

    return "done"


def main():
    files = sorted(glob(os.path.join(PREDS_DIR, "*.nii*")))

    print(f"找到 {len(files)} 个 Stage1 预测文件")
    print(f"输入目录: {PREDS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"NUM_WORKERS = {NUM_WORKERS}")
    print("模式: 低内存并行版，只计算 liver/tumor SDF，background=0")

    if len(files) == 0:
        raise RuntimeError("没有找到任何 .nii / .nii.gz 文件，请检查 PREDS_DIR")

    done = 0
    skipped = 0
    failed = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_case, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating SDF"):
            f = futures[future]

            try:
                result = future.result()

                if result == "skip":
                    skipped += 1
                else:
                    done += 1

            except Exception as e:
                failed.append((f, str(e)))
                print(f"\n失败: {f}")
                print(f"错误: {e}")

    print("\nSDF 生成结束")
    print(f"完成: {done}")
    print(f"跳过: {skipped}")
    print(f"失败: {len(failed)}")

    if failed:
        print("\n失败文件:")
        for path, err in failed:
            print(f"- {path}")
            print(f"  {err}")

        raise RuntimeError(f"共有 {len(failed)} 个文件处理失败")


def send_wechat(msg: str):
    if not WECHAT_TOKEN or WECHAT_TOKEN == "YOUR_TOKEN":
        print("未配置 WECHAT_TOKEN，跳过微信通知")
        return

    try:
        url = f"https://www.autodl.com/api/v1/wechat/message/push?token={WECHAT_TOKEN}"
        data = {
            "title": msg,
            "name": msg,
            "content": msg,
        }
        requests.post(url, json=data, timeout=10)
    except Exception as e:
        print(f"微信通知失败: {e}")


if __name__ == "__main__":
    success = False

    try:
        main()
        success = True
        send_wechat("计算完成")

    except Exception as e:
        send_wechat(f"计算失败")
        raise

    finally:
        if AUTO_SHUTDOWN and success:
            print("10 秒后自动关机...")
            time.sleep(10)
            os.system("/usr/bin/shutdown")

