import os
from glob import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    EnsureTyped, SaveImaged, AsDiscreted, Invertd
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from tqdm import tqdm

# 配置路径与参数
PATCH_SIZE = (192, 192, 192)
PIXDIM = (1.0, 1.0, 1.5)
FEAT_SIZE = 32
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.expanduser("~/autodl-tmp/LiTs")
STAGE1_MODEL_PATH = os.path.expanduser("~/autodl-tmp/lits_heavy_train_192/best_heavy_model.pth")
OUTPUT_DIR = os.path.expanduser("~/autodl-tmp/lits_stage1_preds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 获取所有图像文件
all_imgs = glob(os.path.join(ROOT_DIR, "**", "volume-*.nii*"), recursive=True)
data = [{"image": img} for img in all_imgs]

# 2. 定义数据加载与预处理
pre_trans = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=PIXDIM, mode=("bilinear")),
    Orientationd(keys=["image"], axcodes="RAS"),
    EnsureTyped(keys=["image"]),
])

# 3. 定义后处理与保存操作
# 注意：我们要把预测结果逆变换回原始图像的分辨率和空间方向，保证能完全对齐
post_trans = Compose([
    EnsureTyped(keys="pred"),
    AsDiscreted(keys="pred", argmax=True),
    Invertd(
        keys="pred",
        transform=pre_trans,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    ),
    SaveImaged(
        keys="pred",
        meta_keys="pred_meta_dict",
        output_dir=OUTPUT_DIR,
        output_postfix="", 
        separate_folder=False,
        resample=False,
        output_ext=".nii.gz" # 统一保存为 nii.gz
    )
])

dataset = Dataset(data=data, transform=pre_trans)
loader = DataLoader(dataset, batch_size=1, num_workers=4)

# 4. 构建与加载模型 (与Source 2中一致的模型结构)
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

# 加载 Stage 1 权重
ckpt = torch.load(STAGE1_MODEL_PATH, map_location=DEVICE)
# 如果保存的是字典形式，提取模型权重
if "model" in ckpt:
    model.load_state_dict(ckpt["model"], strict=False)
elif "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)

model.eval()

# 5. 开始推理并生成掩码
print(f"开始生成预测掩码，共 {len(data)} 个文件，保存至 {OUTPUT_DIR}")
with torch.no_grad():
    for batch_data in tqdm(loader):
        img = batch_data["image"].to(DEVICE)
        
        def predictor(x):
            y = model(x)
            if torch.is_tensor(y) and y.ndim == 6: return y[:, 0]
            elif isinstance(y, (list, tuple)): return y[0]
            return y
            
        with torch.amp.autocast("cuda"):
            out = sliding_window_inference(img, PATCH_SIZE, 2, predictor, overlap=0.5)
            
        batch_data["pred"] = out
        
        # 解包 batch 并应用后处理（这里会自动触发 SaveImaged 将文件写入硬盘）
        for item in decollate_batch(batch_data):
            post_trans(item)

print("生成完毕！你可以运行级联训练代码了。")