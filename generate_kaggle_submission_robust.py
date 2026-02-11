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

# ==========================================
# 🔧 用户配置区域 (修改这里即可)
# ==========================================
class CONFIG:
    # 1. 权重文件路径 (请修改为您当前最新的 .pth 文件路径)
    CHECKPOINT_PATH = '/home/swj/dd1/LGBPN-master1/output/your_SIDD_model_name/checkpoint/your_SIDD_model_name_020.pth' 
    
    # 2. 模型参数设置 (根据您当前训练的设定调整)
    MODEL_ARGS = {
        'pd_a': 5, 
        'pd_b': 2, 
        'pd_pad': 2, 
        'R3': True, 
        'R3_T': 8, 
        'R3_p': 0.16, 
        'SIDD': True,
        'bsn': 'My_BSN', 
        'in_ch': 3, 
        'bsn_base_ch': 128, 
        'bsn_num_module': 9
    }
    
    # 3. 数据集 URL (SIDD Benchmark)
    DATA_URL = 'http://130.63.97.225/share/SIDD_Blocks/BenchmarkNoisyBlocksSrgb.mat'
    INPUT_FILE_NAME = 'BenchmarkNoisyBlocksSrgb.mat'
    
    # 4. 是否使用 CUDA
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 🚀 1. 环境与模型导入
# ==========================================
sys.path.append('.') 
# ⚠️ 请确保当前目录下有 model 文件夹，且包含 APBSN12.py
try:
    from src.model.APBSN import BSN 
except ImportError:
    print("❌ 错误: 找不到 'model.APBSN12'。请确保当前目录包含 model 文件夹。")
    sys.exit(1)

print(f"Using device: {CONFIG.DEVICE}")

# ==========================================
# 🛠️ 2. 初始化模型与加载权重
# ==========================================
def load_model():
    print("正在初始化 BSN 模型...")
    # 使用配置字典解包参数
    model = BSN(**CONFIG.MODEL_ARGS).to(CONFIG.DEVICE)

    print(f"正在加载权重: {CONFIG.CHECKPOINT_PATH}")
    if not os.path.exists(CONFIG.CHECKPOINT_PATH):
        print(f"❌ 错误: 找不到权重文件: {CONFIG.CHECKPOINT_PATH}")
        print("   -> 请在代码顶部的 CONFIG 中修改 CHECKPOINT_PATH")
        sys.exit(1)

    try:
        checkpoint = torch.load(CONFIG.CHECKPOINT_PATH, map_location=CONFIG.DEVICE)
        
        # 兼容不同的保存格式
        if 'model_weight' in checkpoint:
            state_dict = checkpoint['model_weight']['denoiser']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 清洗 Key (去除 module. 前缀，处理多卡训练权重的常见问题)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): 
                name = name[7:] 
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 模型权重加载成功 (Strict Mode)！")
        
    except Exception as e:
        print(f"⚠️ Strict加载失败: {e}")
        print("   -> 尝试非严格模式加载 (strict=False)...")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ 模型权重加载成功 (Non-Strict Mode)！")
        except Exception as e2:
            print(f"❌ 最终加载失败: {e2}")
            sys.exit(1)

    model.eval()
    return model

model = load_model()

# ==========================================
# ⚡ 3. 核心处理函数
# ==========================================
def my_srgb_denoiser(x, model):
    """
    x: (256, 256, 3) uint8 numpy array
    return: (256, 256, 3) uint8 numpy array
    """
    # 1. 预处理
    # ⚠️ 注意: 遵循您的旧代码逻辑，这里转 float 但不除以 255.0
    # 如果您更改了训练逻辑（例如变成了 0-1 输入），请在这里改为 x.astype(np.float32) / 255.0
    img = x.astype(np.float32) 
    img = np.transpose(img, (2, 0, 1)) # (H,W,C) -> (C,H,W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(CONFIG.DEVICE) # (1,C,H,W)

    # 2. 推理
    with torch.no_grad():
        output_tensor = model.denoise(img_tensor)

    # 3. 后处理
    output = output_tensor.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0)) # (C,H,W) -> (H,W,C)
    
    # 截断与量化
    output = np.clip(output, 0, 255)
    output = output.round().astype(np.uint8)
    
    return output

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

# ==========================================
# 🔄 4. 主执行流程
# ==========================================
def main():
    # 4.1 准备数据
    if not os.path.exists(CONFIG.INPUT_FILE_NAME):
        print(f'正在下载测试数据 {CONFIG.INPUT_FILE_NAME} ...')
        try:
            wget.download(CONFIG.DATA_URL, CONFIG.INPUT_FILE_NAME)
            print("\n下载完成。")
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            return
    else:
        print(f'检测到数据文件: {CONFIG.INPUT_FILE_NAME}')

    print("正在读取 .mat 文件 (这可能需要几秒钟)...")
    try:
        inputs = scipy.io.loadmat(CONFIG.INPUT_FILE_NAME)['BenchmarkNoisyBlocksSrgb']
        print(f'数据维度: {inputs.shape} (Blocks, Images, H, W, C)')
    except Exception as e:
        print(f"❌ 读取 mat 文件失败: {e}")
        return

    # 4.2 开始去噪循环
    output_blocks_base64string = []
    total_samples = inputs.shape[0] * inputs.shape[1]
    count = 0
    start_time = time.time()

    print(f"🔥 开始处理 {total_samples} 个图像块...")

    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            in_block = inputs[i, j, :, :, :]
            
            # ---> 核心去噪 <---
            out_block = my_srgb_denoiser(in_block, model)
            
            # 自检 (仅前3张)
            if count < 3:
                diff = np.abs(in_block.astype(float) - out_block.astype(float)).mean()
                print(f"   [自检] ID {count}: Mean Diff = {diff:.4f} (预期应有显著差异)")
            
            # 编码
            out_block_base64string = array_to_base64string(out_block)
            output_blocks_base64string.append(out_block_base64string)
            
            count += 1
            
            # 进度打印
            if count % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / count
                remain_time = avg_time * (total_samples - count) / 60
                sys.stdout.write(f"\r进度: {count}/{total_samples} | 耗时: {elapsed:.0f}s | 剩余: {remain_time:.1f} min")
                sys.stdout.flush()

    print("\n处理完成！")

    # 4.3 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_df = pd.DataFrame()
    output_df['ID'] = np.arange(len(output_blocks_base64string))
    output_df['BLOCK'] = output_blocks_base64string

    # 文件名
    file_psnr = f'SubmitSrgb_PSNR_{timestamp}.csv'
    file_ssim = f'SubmitSrgb_SSIM_{timestamp}.csv'

    print(f"正在保存 CSV 文件...")
    output_df.to_csv(file_psnr, index=False)
    output_df.to_csv(file_ssim, index=False)

    print('=' * 60)
    print(f'✅ 成功生成提交文件:')
    print(f' 📂 {file_psnr}')
    print(f' 📂 {file_ssim}')
    print('=' * 60)

if __name__ == "__main__":
    main()