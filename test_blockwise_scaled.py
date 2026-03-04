import torch
import gemm_int8

# 设置矩阵维度
M, K, N = 1024, 2048, 2560
block_size = 128

# 检查维度是否能被 block_size 整除
assert M % block_size == 0, f"M ({M}) must be divisible by block_size ({block_size})"
assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"
assert N % block_size == 0, f"N ({N}) must be divisible by block_size ({block_size})"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建 int8 输入矩阵
A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
B_int8_t = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)

# 创建 scale 张量 (每个 128x128 block 一个 scale)
scale_a = torch.rand(M // block_size, K // block_size, dtype=torch.float32, device=device) * 0.1
scale_b = torch.rand(N // block_size, K // block_size, dtype=torch.float32, device=device) * 0.1

print(f"\n矩阵维度:")
print(f"  A: {A_int8.shape}")
print(f"  B: {B_int8_t.shape}")
print(f"  scale_a: {scale_a.shape}")
print(f"  scale_b: {scale_b.shape}")

# 测试新的 blockwise scaled GEMM
print("\n测试 blockwise scaled GEMM...")
try:
    C_scaled = gemm_int8.matmul_blockwise_scaled(A_int8, B_int8_t, scale_a, scale_b)
    print(f"✓ 成功! 输出形状: {C_scaled.shape}, dtype: {C_scaled.dtype}")
    print(f"  输出范围: [{C_scaled.min().item():.4f}, {C_scaled.max().item():.4f}]")
    print(f"  输出均值: {C_scaled.mean().item():.4f}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 对比原始的 matmul (使用平均 scale)
print("\n对比原始 matmul...")
try:
    mean_scale = scale_a.mean().item() * scale_b.mean().item()
    C_original = gemm_int8.matmul(A_int8, B_int8_t, alpha=mean_scale)
    print(f"✓ 原始 matmul 输出形状: {C_original.shape}")
    print(f"  输出范围: [{C_original.min().item():.4f}, {C_original.max().item():.4f}]")
    print(f"  输出均值: {C_original.mean().item():.4f}")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试完成!")
