import torch
from thop import profile
import sys
import os

sys.path.append('.') 
import src.model.DBSNl as DBSNl_Module 
from src.model.DBSNl import LGBPN 

# ==========================================
# 🛡️ 补丁 1: Mock DataParallel (强制 CPU)
# ==========================================
def mock_data_parallel(module, inputs, device_ids=None, output_device=None):
    """
    欺骗模型：假装自己在做多卡并行，但实际上强制把数据搬到 CPU 上单线程跑。
    这样既解决了 OOM (显存不够)，也解决了 Device Mismatch。
    """
    target_device = torch.device('cpu')
    
    # 1. 确保 module 在 CPU
    if hasattr(module, 'to'):
        module = module.to(target_device)
    
    # 2. 递归把 inputs 搬到 CPU
    def move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(target_device)
        elif isinstance(obj, (list, tuple)):
            return [move_to_device(x) for x in obj]
        return obj
    
    inputs = move_to_device(inputs)

    # 3. 直接运行
    if isinstance(inputs, (list, tuple)):
        return module(*inputs)
    return module(inputs)

# 注入补丁
DBSNl_Module.P.data_parallel = mock_data_parallel

# ==========================================
# 🛡️ 补丁 2: Mock device_count
# ==========================================
# 虽然我们在 CPU 上，但模型逻辑依赖 n_GPUs 来决定切片数量。
# 我们假装有 1 张卡，让代码进入正确的循环逻辑 (range(0,4,1))。
torch.cuda.device_count = lambda: 1

print("✅ 已切换至 CPU 模式，避免 CUDA OOM。")
print("✅ 已应用逻辑补丁：Mock DataParallel + Mock DeviceCount=1")


def main():
    # ================= 1. 环境设置 =================
    # 强制使用 CPU
    device = torch.device('cpu')
    print(f"🚀 Using device: {device}")

    # ================= 2. 配置模型 =================
    model = LGBPN(
        in_ch=3, 
        out_ch=3, 
        base_ch=128,    
        num_module=9,   
        br2_blc=6,      
        SIDD=True
    ).to(device)
    
    model.eval()

    # ================= 3. 设定输入 =================
    # 保持 256x256 的标准学术对比尺寸
    input_size = 256
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)

    print(f"🔥 正在计算 FLOPs (输入尺寸: {input_size}x{input_size})...")
    print("⏳这可能需要几秒钟，请耐心等待...")

    # ================= 4. 计算 =================
    try:
        flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
        
        # ================= 5. 输出结果 =================
        print("\n" + "=" * 40)
        print(f"📊 统计结果 (LGBPN)")
        print(f"   输入分辨率: {input_size} x {input_size}")
        print("-" * 40)
        print(f"   Params (参数量): {params / 1e6 :.4f} M (百万)")
        print(f"   FLOPs  (计算量): {flops / 1e9 :.4f} G (十亿)")
        print("=" * 40 + "\n")
        print("✅ 计算成功！")
        
    except Exception as e:
        print(f"\n❌ 计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()