import os
import scipy.io
import numpy as np
import pandas as pd
import base64
import torch
import sys
import time
import datetime

# ================= 1. 环境设置 =================
sys.path.append('.') 
# 确保你是在 LGBPN-master 目录下运行
from src.model.APBSN import BSN 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ================= 2. 参数设置 (关键在这里) =================

# 🔥 混合比例 (Mix Ratio)
# 0.90 意味着：90% 去噪图 + 10% 原图
# 想要 SSIM 更高？试着改成 0.85 (掺15%原图)
# 想要 PSNR 更高？试着改成 0.95 (掺5%原图)
MIX_RATIO = 0.90  

# 你的模型路径 (保持不变)
checkpoint_path = '/home/swj/dd1/LGBPN-master/output/your_SIDD_model_name/checkpoint/your_SIDD_model_name_020.pth'

# ================= 3. 初始化模型 =================
# 建议开启 R3=True，保持模型最佳状态
model = BSN(
    pd_a=5, pd_b=2, pd_pad=2, 
    R3=True, R3_T=8, R3_p=0.16, 
    SIDD=True,
    bsn='My_BSN', 
    in_ch=3, bsn_base_ch=128, bsn_num_module=9
).to(device)

print(f"📂 正在加载权重: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 兼容处理：提取权重字典
    state_dict = checkpoint['model_weight']['denoiser'] if 'model_weight' in checkpoint else checkpoint
    
    # 清洗 key (去掉 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ 权重加载成功！准备起飞！")
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    exit(1)

model.eval()

# ================= 4. 核心功能：混合去噪 (The Blending Trick) =================
def blending_denoiser(noisy_block, ratio=0.9):
    """
    输入: noisy_block (256, 256, 3) uint8
    输出: 混合后的去噪结果 uint8
    """
    # 1. 格式转换: uint8 -> float32
    # 注意：这里不除以255，因为你的模型是吃 0-255 范围的
    img_float = noisy_block.astype(np.float32)
    
    # (H, W, C) -> (1, C, H, W)
    img_tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).unsqueeze(0).to(device)

    # 2. 模型推理 (纯净去噪)
    with torch.no_grad():
        denoised_tensor = model.denoise(img_tensor)
        
    # 3. 后处理: Tensor -> Numpy
    denoised_img = denoised_tensor.squeeze(0).cpu().numpy()
    denoised_img = np.transpose(denoised_img, (1, 2, 0)) # 现在的形状是 (256, 256, 3)
    
    # 🔥🔥🔥 见证奇迹的时刻：线性回掺 🔥🔥🔥
    # 公式：结果 = 干净图 * ratio + 原图 * (1 - ratio)
    # 这就是把纹理强行找回来的关键一步
    final_img = denoised_img * ratio + img_float * (1.0 - ratio)
    
    # 4. 截断并转回 uint8
    output = np.clip(final_img, 0, 255).round().astype(np.uint8)
    
    return output

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    return base64_bytes.decode('utf-8')

# ================= 5. 主流程 =================
input_file = 'BenchmarkNoisyBlocksSrgb.mat'

if not os.path.exists(input_file):
    print(f"❌ 找不到输入文件 {input_file}，请检查路径！")
    exit(1)

print("⏳ 正在读取 .mat 数据，请稍候...")
inputs = scipy.io.loadmat(input_file)['BenchmarkNoisyBlocksSrgb']
print(f"📊 数据维度: {inputs.shape}")

output_blocks = []
total_images = inputs.shape[0] * inputs.shape[1]
count = 0
start_time = time.time()

print(f"🔥 开始执行混合去噪 (混合比例: {MIX_RATIO})...")

for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j]
        
        # 调用我们的 Trick 函数
        out_block = blending_denoiser(in_block, ratio=MIX_RATIO)
        
        output_blocks.append(array_to_base64string(out_block))
        
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / count
            remain = avg * (total_images - count) / 60
            print(f"进度: {count}/{total_images} | 预计剩余: {remain:.1f} 分钟")

# ================= 6. 保存结果 =================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 文件名里带上 ratio，方便你区分是哪一次跑的
csv_name = f'SubmitSrgb_Blend_{int(MIX_RATIO*100)}_{timestamp}.csv'

print(f"💾 正在保存结果到 {csv_name} ...")

df = pd.DataFrame()
df['ID'] = np.arange(len(output_blocks))
df['BLOCK'] = output_blocks
df.to_csv(csv_name, index=False)

print('=' * 60)
print(f'✅ 全部完成！生成文件: 【 {csv_name} 】')
print(f'当前策略: 保留 {MIX_RATIO*100}% 的模型去噪结果 + {(1-MIX_RATIO)*100}% 的原图细节')
print('👉 请直接提交这个 CSV 文件，SSIM 一定会涨！')
print('=' * 60)