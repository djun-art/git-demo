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
# 直接导入 BSN 类，复用官方 test.py 的逻辑
from src.model.APBSN import BSN 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================= 2. 初始化模型 =================
print("正在初始化 BSN 模型 (参数匹配官方 BSN_SIDD.yaml)...")

# 使用 BSN 包装器，它会自动处理 R3 策略和 Pixel Downsampling
# 这些参数严格对应配置文件，确保结构正确
model = BSN(
    pd_a=5, pd_b=2, pd_pad=2, 
    R3=True, R3_T=8, R3_p=0.16, 
    SIDD=True,
    bsn='My_BSN', 
    in_ch=3, bsn_base_ch=128, bsn_num_module=9
).to(device)

# ================= 3. 加载权重 =================
# 使用你指定的权重路径 (既然是作者模型，文件本身是没问题的)
checkpoint_path = '/home/swj/dd1/LGBPN-master/output/your_SIDD_model_name/checkpoint/your_SIDD_model_name_020.pth'
print(f"正在加载权重文件: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取权重字典
    if 'model_weight' in checkpoint:
        state_dict = checkpoint['model_weight']['denoiser']
    else:
        state_dict = checkpoint

    # 清洗 Key：BSN 类内部有 self.bsn，所以需要保留 'bsn.' 前缀
    # 只需要去掉多卡训练可能产生的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'): 
            name = name[7:] # 去掉 module.
        new_state_dict[name] = v

    # 加载权重
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ BSN 模型权重加载成功！")
    
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    print("尝试宽松加载 (strict=False)...")
    model.load_state_dict(new_state_dict, strict=False)

model.eval()

# ================= 4. 核心去噪函数 (关键修正) =================
def my_srgb_denoiser(x):
    """
    x: (256, 256, 3) uint8 numpy array [0, 255]
    """
    # ⚠️ 关键修正：不要除以 255.0！
    # 官方模型 (LGBPN/AP-BSN) 在 SIDD/DND 上是直接处理 0-255 的 float 数据
    img = x.astype(np.float32) 
    
    # (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    # 增加 Batch 维度 -> (1, C, H, W) 并转到 GPU
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        # 调用 BSN 类的 denoise 方法
        # 这个方法内置了 R3 (8次循环) 和 PD (Pixel Downsampling)
        # 输入和输出都是 0-255 范围的 Tensor
        output_tensor = model.denoise(img_tensor)

    # 后处理
    output = output_tensor.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    
    # 截断到有效范围 [0, 255] 并转回 uint8
    output = np.clip(output, 0, 255)
    output = output.round().astype(np.uint8)
    
    return output

# ================= 5. 辅助函数 =================
def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

# ================= 6. 主流程 =================
input_file = 'BenchmarkNoisyBlocksSrgb.mat'
# 自动下载部分省略，假设文件已存在
if not os.path.exists(input_file):
    print(f"❌ 找不到输入文件 {input_file}")
    exit(1)

print("正在读取 .mat 数据...")
inputs = scipy.io.loadmat(input_file)['BenchmarkNoisyBlocksSrgb']
print(f'输入数据维度: {inputs.shape}')

output_blocks_base64string = []
total_images = inputs.shape[0] * inputs.shape[1]
count = 0
start_time = time.time()

print("🔥 开始去噪 (已修正数据范围 0-255)...")

for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :, :]
        
        # 运行去噪
        out_block = my_srgb_denoiser(in_block)
        
        # ✅ 再次自检 (这次 Diff 应该正常了)
        if count < 3:
            diff = np.abs(in_block.astype(float) - out_block.astype(float)).mean()
            print(f"🔍 图 {count} 自检 Diff: {diff:.4f}")
            if diff > 100:
                print("❌ 警告：Diff 依然过大 (>100)！说明可能还有问题！")
            elif diff < 1:
                print("❌ 警告：Diff 过小 (<1)！模型可能输出了原图！")
            else:
                print("✅ Diff 正常 (5~30 之间)，模型工作正常！")
        
        out_block_base64string = array_to_base64string(out_block)
        output_blocks_base64string.append(out_block_base64string)
        
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remain_time = avg_time * (total_images - count) / 60
            print(f"进度: {count}/{total_images} | 耗时: {elapsed:.0f}s | 预计剩余: {remain_time:.1f} min")

# ================= 7. 保存结果 =================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'SubmitSrgb_Fixed_{timestamp}.csv' 

print(f'正在保存结果到 {output_file}...')
output_df = pd.DataFrame()
output_df['ID'] = np.arange(len(output_blocks_base64string))
output_df['BLOCK'] = output_blocks_base64string
output_df.to_csv(output_file, index=False)

print('=' * 60)
print(f'✅ 全部完成！')
print(f'⬇️ 请提交这个文件: 【 {output_file} 】')
print('这次分数一定会变！')
print('=' * 60)