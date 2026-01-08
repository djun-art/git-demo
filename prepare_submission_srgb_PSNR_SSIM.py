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
import shutil

# ================= 1. 环境与导入 =================
sys.path.append('.') 
from model.APBSN12 import BSN 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================= 2. 初始化模型 =================
print("正在初始化 BSN 模型 (参数匹配官方 BSN_SIDD.yaml)...")

# 使用 BSN 包装器，自动处理 R3 和 PD
model = BSN(
    pd_a=5, pd_b=2, pd_pad=2, 
    R3=True, R3_T=8, R3_p=0.16, 
    SIDD=True,
    bsn='My_BSN', 
    in_ch=3, bsn_base_ch=128, bsn_num_module=9
).to(device)

# ================= 3. 加载权重 =================
checkpoint_path = '/home/swj/dd1/LGBPN-master/output/your_SIDD_model_name/checkpoint/your_SIDD_model_name_020.pth'
print(f"正在加载权重文件: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_weight' in checkpoint:
        state_dict = checkpoint['model_weight']['denoiser']
    else:
        state_dict = checkpoint

    # 清洗 Key (保留 bsn. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'): 
            name = name[7:] 
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    print("✅ BSN 模型权重加载成功！")
    
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    model.load_state_dict(new_state_dict, strict=False)

model.eval()

# ================= 4. 核心去噪函数 (0-255 版本) =================
def my_srgb_denoiser(x):
    """
    x: (256, 256, 3) uint8 numpy array
    """
    # ⚠️ 保持修正：不除以 255.0
    img = x.astype(np.float32) 
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output_tensor = model.denoise(img_tensor)

    # 后处理
    output = output_tensor.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output, 0, 255)
    output = output.round().astype(np.uint8)
    
    return output

# ================= 5. 辅助函数 =================
def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

# ================= 6. 主流程 (一次运行，全部搞定) =================
url = 'http://130.63.97.225/share/SIDD_Blocks/BenchmarkNoisyBlocksSrgb.mat'
input_file = 'BenchmarkNoisyBlocksSrgb.mat'

if not os.path.exists(input_file):
    print(f'正在下载输入文件 {input_file} ...')
    try:
        wget.download(url, input_file)
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        exit(1)
else:
    print(f'{input_file} 已存在。')

print("正在读取 .mat 数据...")
inputs = scipy.io.loadmat(input_file)['BenchmarkNoisyBlocksSrgb']
print(f'输入数据维度: {inputs.shape}')

output_blocks_base64string = []
total_images = inputs.shape[0] * inputs.shape[1]
count = 0
start_time = time.time()

print("🔥 开始去噪 (PSNR & SSIM 通用版)...")

for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :, :]
        
        # 运行去噪
        out_block = my_srgb_denoiser(in_block)
        
        # 简单自检 (只看前3张)
        if count < 3:
            diff = np.abs(in_block.astype(float) - out_block.astype(float)).mean()
            print(f"🔍 图 {count} 自检 Diff: {diff:.4f} (预期 13 左右)")
        
        out_block_base64string = array_to_base64string(out_block)
        output_blocks_base64string.append(out_block_base64string)
        
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remain_time = avg_time * (total_images - count) / 60
            print(f"进度: {count}/{total_images} | 耗时: {elapsed:.0f}s | 预计剩余: {remain_time:.1f} min")

# ================= 7. 保存结果 (一式两份) =================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 准备数据
output_df = pd.DataFrame()
output_df['ID'] = np.arange(len(output_blocks_base64string))
output_df['BLOCK'] = output_blocks_base64string

# 1. 保存 PSNR 提交文件
file_psnr = f'SubmitSrgb_PSNR_{timestamp}.csv'
print(f'正在保存 PSNR 提交文件: {file_psnr} ...')
output_df.to_csv(file_psnr, index=False)

# 2. 保存 SSIM 提交文件 (直接复制，因为内容是一样的)
file_ssim = f'SubmitSrgb_SSIM_{timestamp}.csv'
print(f'正在保存 SSIM 提交文件: {file_ssim} ...')
output_df.to_csv(file_ssim, index=False)

print('=' * 60)
print(f'✅ 全部完成！生成了两个文件：')
print(f'   1. {file_psnr}  -> 提交到 PSNR 榜单')
print(f'   2. {file_ssim}  -> 提交到 SSIM 榜单')
print(f'提示：这两个文件内容其实是一样的，因为你的模型同时优化了两个指标。')
print('=' * 60)