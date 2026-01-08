import wget
import os
import scipy.io
import numpy as np
import pandas as pd
import base64
import torch
import sys
import time
import datetime

# ================= 1. 环境与导入 =================
sys.path.append('.') 
from src.model.APBSN import BSN 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================= 2. 初始化模型 =================
# ✅ 保持和训练时一致的配置
model = BSN(
    pd_a=5, pd_b=2, pd_pad=2, 
    R3=True, R3_T=8, R3_p=0.16, 
    SIDD=True,
    bsn='My_BSN', 
    in_ch=3, bsn_base_ch=128, bsn_num_module=9
).to(device)

# ================= 3. 加载权重 =================
# 请确认这里是你那个“PSNR 37.68”的好模型路径
checkpoint_path = '/home/swj/dd1/LGBPN-master/output/your_SIDD_model_name/checkpoint/your_SIDD_model_name_020.pth'
print(f"正在加载权重: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_weight']['denoiser'] if 'model_weight' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ 权重加载成功！")
except Exception as e:
    print(f"⚠️ 权重加载异常: {e}")
    exit(1)

model.eval()

# ================= 4. 定义辅助函数 (翻转/旋转) =================
def rot_hflip_img(img: torch.Tensor, rot_times: int = 0, hflip: int = 0):
    '''
    核心 TTA 函数：对 Tensor 进行旋转和翻转
    '''
    # 0 = no flip, 1 = horizontal flip
    b = 0 if len(img.shape) == 3 else 1
    
    # 1. 先处理翻转
    if hflip % 2 == 1:
        img = img.flip(b + 2) # 水平翻转 (W维度)

    # 2. 再处理旋转
    if rot_times % 4 == 0:
        return img
    elif rot_times % 4 == 1:
        return img.transpose(b + 1, b + 2).flip(b + 1) # 90度
    elif rot_times % 4 == 2:
        return img.flip(b + 1).flip(b + 2) # 180度
    elif rot_times % 4 == 3:
        return img.transpose(b + 1, b + 2).flip(b + 2) # 270度
    
    return img

def rot_hflip_img_inverse(img: torch.Tensor, rot_times: int = 0, hflip: int = 0):
    '''
    逆操作：把旋转翻转后的图还原回来
    '''
    b = 0 if len(img.shape) == 3 else 1
    
    # 1. 先逆旋转
    # 顺时针转 k 次的逆操作是顺时针转 4-k 次
    if rot_times % 4 == 1:
        img = img.transpose(b + 1, b + 2).flip(b + 2) # 逆90 = 转270
    elif rot_times % 4 == 2:
        img = img.flip(b + 1).flip(b + 2) # 逆180
    elif rot_times % 4 == 3:
        img = img.transpose(b + 1, b + 2).flip(b + 1) # 逆270 = 转90
        
    # 2. 再逆翻转 (翻转的逆操作还是翻转)
    if hflip % 2 == 1:
        img = img.flip(b + 2)
        
    return img

# ================= 5. 核心去噪函数 (带 Self-Ensemble) =================
def my_srgb_denoiser(x):
    """
    x: (256, 256, 3) uint8 numpy array [0, 255]
    """
    # 1. 预处理
    img = x.astype(np.float32) 
    img = np.transpose(img, (2, 0, 1))
    # (1, C, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # 2. 推理 (Self-Ensemble / TTA)
    # 这是一个极其强大的提分技巧：将图像进行 8 种几何变换，分别去噪，再取平均
    with torch.no_grad():
        result = torch.zeros_like(img_tensor)
        
        # 遍历 4 个旋转角度 * 2 个翻转状态 = 8 种情况
        for i in range(8):
            rot = i % 4
            flip = i // 4
            
            # A. 变换输入
            aug_input = rot_hflip_img(img_tensor, rot_times=rot, hflip=flip)
            
            # B. 去噪 (包含 R3 策略)
            aug_output = model.denoise(aug_input)
            
            # C. 还原输出
            out_inverse = rot_hflip_img_inverse(aug_output, rot_times=rot, hflip=flip)
            
            result += out_inverse
            
        # D. 取平均
        output_tensor = result / 8.0

    # 3. 后处理
    output = output_tensor.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output, 0, 255).round().astype(np.uint8)
    
    return output

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    return base64_bytes.decode('utf-8')

# ================= 6. 主流程 =================
input_file = 'BenchmarkNoisyBlocksSrgb.mat'
if not os.path.exists(input_file):
    print("请先下载输入文件！")
    exit(1)

inputs = scipy.io.loadmat(input_file)['BenchmarkNoisyBlocksSrgb']
print(f"数据加载完毕: {inputs.shape}")

output_blocks = []
total = inputs.shape[0] * inputs.shape[1]
count = 0
start = time.time()

print("🔥 开始强力去噪 (Self-Ensemble + R3)... SSIM 必涨！")

for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j]
        out_block = my_srgb_denoiser(in_block)
        output_blocks.append(array_to_base64string(out_block))
        
        count += 1
        if count % 50 == 0:
            elapsed = time.time() - start
            print(f"进度: {count}/{total} | 耗时: {elapsed:.1f}s")

# 保存
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f'SubmitSrgb_SelfEnsemble_{timestamp}.csv'
df = pd.DataFrame({'ID': np.arange(len(output_blocks)), 'BLOCK': output_blocks})
df.to_csv(csv_name, index=False)

print(f"✅ 完成！请提交 {csv_name}")
print("这次 SSIM 应该会显著提升！")