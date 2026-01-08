import torch
import numpy as np
import sys
sys.path.append('.') 
from model.DBSNl import LGBPN 

# 1. 配置和你提交脚本一模一样的路径
checkpoint_path = '/home/swj/dd1/LGBPN-master/output/your_SIDD_model_name/checkpoint/your_SIDD_model_name_020.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 加载模型
print(f"正在检查模型: {checkpoint_path}")
model = LGBPN(in_ch=3, out_ch=3, base_ch=128, num_module=9, head_ch=24, SIDD=True).to(device)
try:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # 模拟你脚本里的 cleaning 逻辑
    state_dict = ckpt['model_weight']['denoiser'] if 'model_weight' in ckpt else ckpt
    new_state_dict = {k.replace('module.', '').replace('bsn.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 3. 灵魂拷问：它真的在去噪吗？
model.eval()
with torch.no_grad():
    # 造一张纯随机噪点图 (模拟输入)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 跑一次模型
    output = model(dummy_input)
    
    # 【关键】检查输出和输入是否一模一样
    diff = torch.abs(output - dummy_input).mean().item()
    
    print("-" * 30)
    print(f"输入输出平均差异值: {diff:.6f}")
    if diff < 1e-4:
        print("🚨 实锤了！这个模型是'哑巴'！它直接输出了原图！")
        print("原因：你加载的这个 checkpoint 可能根本没训练，或者是一个初始化的空壳。")
    else:
        print("🟢 模型有反应，输出和输入不一样。")
        print(f"差异值 {diff} 说明它确实在修改图像。")