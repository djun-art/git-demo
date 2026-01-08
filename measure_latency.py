import torch
import torch.nn as nn
import time
import numpy as np
import sys

# 导入你的模型
sys.path.append('.') 
from src.model.DBSNl import LGBPN

# ================= 配置区域 =================
# 建议与训练时的 patch size 保持一致或使用标准测试尺寸
INPUT_SIZE = (1, 3, 256, 256) 
BASE_CH = 32   # 确保这里和你 FLOPs 统计时用的通道数一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOP_TIMES = 100  # 循环次数，越多越准
WARM_UP = 20      # 预热次数
# ===========================================

def measure_inference_speed(model, input_tensor, name="Model"):
    model.eval()
    
    # 1. GPU 预热 (Warm-up)
    # 刚开始运行通过 GPU 时会有各种初始化开销，必须先跑几十次让它“热”起来
    print(f"🔥 [{name}] 正在预热 GPU...")
    with torch.no_grad():
        for _ in range(WARM_UP):
            _ = model(input_tensor)
    
    # 2. 正式测速
    print(f"⏱️  [{name}] 开始测速 (循环 {LOOP_TIMES} 次)...")
    
    # 定义 CUDA 事件用于精确计时
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((LOOP_TIMES, 1))
    
    with torch.no_grad():
        for i in range(LOOP_TIMES):
            starter.record()
            _ = model(input_tensor)
            ender.record()
            
            # 等待 GPU 完成所有任务 (同步)
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 毫秒
            timings[i] = curr_time

    # 3. 统计结果
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    fps = 1000 / mean_time
    
    print(f"✅ [{name}] 结果:")
    print(f"    平均延迟 (Latency): {mean_time:.4f} ms")
    print(f"    吞吐量 (FPS):       {fps:.2f} images/sec")
    print("-" * 50)
    return mean_time

def main():
    if str(DEVICE) == 'cpu':
        print("⚠️ 警告: 未检测到 GPU，将在 CPU 上运行。SimpleGate 的优势在 CPU 上可能不如 GPU 明显。")
    
    # 创建输入张量
    input_tensor = torch.randn(INPUT_SIZE).to(DEVICE)

    # ----------------------------------------------------
    # 模型 1: Baseline (无 SimpleGate, 无 SKFusion)
    # ----------------------------------------------------
    model_baseline = LGBPN(
        in_ch=3, out_ch=3, base_ch=BASE_CH, 
        num_module=9, group=1, head_ch=24, br2_blc=6, SIDD=True,
        use_sk_fusion=False,    # <--- 关闭
        use_simplegate=False    # <--- 关闭 (使用 GELU)
    ).to(DEVICE)
    
    latency_base = measure_inference_speed(model_baseline, input_tensor, name="Baseline (GELU + Cat)")

    # ----------------------------------------------------
    # 模型 2: Proposed (有 SimpleGate, 有 SKFusion)
    # ----------------------------------------------------
    model_proposed = LGBPN(
        in_ch=3, out_ch=3, base_ch=BASE_CH, 
        num_module=9, group=1, head_ch=24, br2_blc=6, SIDD=True,
        use_sk_fusion=True,     # <--- 开启
        use_simplegate=True     # <--- 开启 (使用 SimpleGate)
    ).to(DEVICE)
    
    latency_prop = measure_inference_speed(model_proposed, input_tensor, name="Proposed (SimpleGate + SKFusion)")

    # ----------------------------------------------------
    # 总结对比
    # ----------------------------------------------------
    diff = latency_base - latency_prop
    print("📊【最终对比】")
    if diff > 0:
        print(f"🎉 你的改进方案比 Baseline 快了 {diff:.4f} ms!")
        print(f"🚀 速度提升比例: {diff / latency_base * 100 :.2f}%")
        print("结论: 可以在论文中强调 'Lower Latency' 或 'Better Efficiency'。")
    else:
        print(f"你的方案比 Baseline 慢了 {-diff:.4f} ms。")
        print("原因分析: SKFusion 增加的计算量可能超过了 SimpleGate 节省的时间。")
        print("建议: 尝试单独测试 '仅 SimpleGate' (不加 SKFusion)，那样肯定会快。")

if __name__ == "__main__":
    main()