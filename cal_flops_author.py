import torch
from thop import profile
import sys

sys.path.append('.') 
from src.model.DBSNl import LGBPN, DSPMC_9, DSPMC_21

# ==========================================
# 1. 定义自定义计算函数 (只算非零权重)
# ==========================================
def count_dspmc(m, x, y):
    """
    m: 层对象 (DSPMC_9 或 DSPMC_21)
    x: 输入 (tuple)
    y: 输出 (tensor)
    """
    # 获取输出特征图的尺寸 (B, C, H, W)
    batch_size, out_c, out_h, out_w = y.shape
    
    # 获取输入通道数
    # in_c = m.in_channels 
    
    # 【核心逻辑】：统计 mask 中非零元素的个数
    # m.weight 的形状是 (Out, In, K, K)
    # m.mask 的形状也是 (Out, In, K, K)
    # count_nonzero 得到的就是“有效参数量”
    if hasattr(m, 'mask'):
        # 注意：mask 里的 0 和 1 决定了哪些权重参与运算
        effective_params = torch.count_nonzero(m.mask).item()
    else:
        # 如果万一没有 mask，就按全量算
        effective_params = m.weight.numel()
    
    # FLOPs = 输出像素点数 * 每个像素点需要的有效乘法加法数
    # 每个像素点需要进行的运算次数 = 有效参数量 / 输出通道数 (这是不对的，应该是针对整个卷积核)
    
    # 更正的 FLOPs 公式：
    # 标准卷积 FLOPs ≈ (B * H_out * W_out) * (C_in * K * K * C_out)
    #               = (B * H_out * W_out) * (Total_Params)
    # 我们的有效 FLOPs = (B * H_out * W_out) * (Effective_Params)
    
    total_ops = batch_size * out_h * out_w * effective_params
    
    # 将结果写入 m.total_ops (thop 的规范)
    m.total_ops += torch.DoubleTensor([int(total_ops)])


# ==========================================
# 2. 也是为了防报错的 Mock 补丁
# ==========================================
# 为了防止 DSPMC 内部 forward_chop 里的 DeformConv 报错，我们这里还是用 CPU 跑
import src.model.DBSNl as DBSNl_Module
def mock_data_parallel(module, inputs, device_ids=None, output_device=None):
    if isinstance(inputs, (list, tuple)): return module(*inputs)
    return module(inputs)
DBSNl_Module.P.data_parallel = mock_data_parallel
torch.cuda.device_count = lambda: 1


def main():
    # 强制使用 CPU (计算 FLOPs 不需要 GPU)
    device = torch.device('cpu')
    print(f"🚀 Using device: {device}")

    # ================= 3. 配置模型 =================
    # 注意：这里的参数要和你实际用的保持一致
    # 如果你想复现论文表格里的极低数值，base_ch 可能需要改小 (例如 64)
    model = LGBPN(
        in_ch=3, 
        out_ch=3, 
        base_ch=128,    
        num_module=9,   
        br2_blc=6,      
        SIDD=True
    ).to(device)
    
    model.eval()

    # ================= 4. 设定输入 =================
    input_size = 256
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)

    print(f"🔥 正在计算 '有效' FLOPs (输入尺寸: {input_size}x{input_size})...")

    # ================= 5. 注册自定义算子并计算 =================
    flops, params = profile(
        model, 
        inputs=(input_tensor, ), 
        custom_ops={
            DSPMC_9: count_dspmc,  # 遇到 DSPMC_9 用我们自定义的函数算
            DSPMC_21: count_dspmc  # 遇到 DSPMC_21 用我们自定义的函数算
        },
        verbose=False
    )

    # ================= 6. 输出结果 =================
    print("\n" + "=" * 40)
    print(f"📊 作者同款统计结果 (Effective FLOPs)")
    print("-" * 40)
    print(f"Params (参数量): {params / 1e6 :.4f} M")
    print(f"FLOPs  (计算量): {flops / 1e9 :.4f} G")
    print("-" * 40)
    print("注：此结果已扣除 Mask 中为 0 的无效计算量。")

if __name__ == "__main__":
    main()