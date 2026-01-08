import torch
import torch.nn as nn
from thop import profile
import sys

# 导入你的模型
sys.path.append('.') 
from src.model.DBSNl import LGBPN, DSPMC_9, DSPMC_21
import src.model.DBSNl as DBSNl_Module 

# ==========================================
# 0. 实验配置 (在此处开关你的改进点)
# ==========================================
# 想要测什么版本，就在这里修改 True/False
CONFIG = {
    'use_sk_fusion': False,    # 是否开启 SKFusion
    'use_simplegate':True ,   # 是否开启 SimpleGate
    'base_ch': 32             # 通道数 (建议改小以接近论文量级，例如 32 或 64)
}

# ==========================================
# 1. 定义核心统计函数：只算非零 (Effective)
# ==========================================
def count_dspmc(m, x, y):
    """
    针对 DSPMC 层，只统计 mask 中非零的元素作为有效参数量。
    """
    batch_size, out_c, out_h, out_w = y.shape
    
    # --- 1. 计算有效参数量 (Effective Params) ---
    if hasattr(m, 'mask'):
        # 统计 mask 里有多少个 1
        total_effective_params = torch.count_nonzero(m.mask).item()
        if m.bias is not None:
            total_effective_params += m.bias.numel()
    else:
        total_effective_params = m.weight.numel()
        if m.bias is not None: total_effective_params += m.bias.numel()

    # 覆盖 thop 默认参数量
    m.total_params = torch.DoubleTensor([int(total_effective_params)])

    # --- 2. 计算有效计算量 (Effective FLOPs) ---
    # FLOPs = H * W * Total_Effective_Params
    total_ops = batch_size * out_h * out_w * total_effective_params
    
    m.total_ops += torch.DoubleTensor([int(total_ops)])

# ==========================================
# 2. 必要的 Mock 补丁
# ==========================================
def mock_data_parallel(module, inputs, device_ids=None, output_device=None):
    if isinstance(inputs, (list, tuple)):
        return module(*inputs)
    return module(inputs)

DBSNl_Module.P.data_parallel = mock_data_parallel
torch.cuda.device_count = lambda: 1

def main():
    device = torch.device('cpu')
    
    # ==========================================
    # 3. 配置模型 (已更新适配新接口)
    # ==========================================
    print(f"🚀 正在加载模型... 配置: {CONFIG}")
    
    model = LGBPN(
        in_ch=3, 
        out_ch=3, 
        base_ch=CONFIG['base_ch'],  # 使用配置里的通道数
        num_module=6,
        group=1,
        head_ch=24,    
        br2_blc=6,     
        SIDD=True,
        # --- 新增的参数 ---
        use_sk_fusion=CONFIG['use_sk_fusion'],
        use_simplegate=CONFIG['use_simplegate']
    ).to(device)
    
    model.eval()

    # 输入尺寸
    input_size = 256
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)

    print(f"🔥 正在计算 (Effective Mask 模式)... 输入: {input_size}x{input_size}")

    # ==========================================
    # 4. 执行统计
    # ==========================================
    # 注意：SKFusion 里的 Conv2d 是标准层，thop 会自动识别，不需要 custom_ops
    flops, params = profile(
        model, 
        inputs=(input_tensor, ), 
        custom_ops={
            DSPMC_9: count_dspmc,
            DSPMC_21: count_dspmc
        },
        verbose=False
    )

    # ==========================================
    # 5. 打印结果与分析
    # ==========================================
    print("-" * 60)
    print(f"【统计结果 - {CONFIG['base_ch']} ch】")
    print(f"Params (有效参数量): {params / 1e6 :.4f} M")
    print(f"FLOPs  (有效计算量): {flops / 1e9 :.4f} G")
    print("-" * 60)
    
    # 结果分析提示
    if CONFIG['use_sk_fusion']:
        print("ℹ️  SKFusion 说明: 引入了少量的额外参数 (MLP部分)，你应该能看到 Params 略微增加。")
    
    if CONFIG['use_simplegate']:
        print("ℹ️  SimpleGate 说明: thop 通常不统计激活函数(GELU)的FLOPs，因此你可能看到 FLOPs 数值变化极小或没变。")
        print("    但 SimpleGate 的优势在于 GPU 上的实际推理速度 (Latency)，而非单纯的乘加运算次数。")

if __name__ == "__main__":
    main()