import torch
import gemm_int8

# 设置矩阵维度
M, K, N = 8192, 2048, 2560

# 预热 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建 bf16 输入矩阵
A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

# 创建 int8 输入矩阵
A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)

# 创建 gemm_int8 输入矩阵 (注意: gemm_int8.matmul 计算 x @ y.t(), 所以 y 的形状是 (N, K))
B_int8_t = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)

# 创建 blockwise scale 张量 (128x128 blocks)
block_size = 128
scale_a = torch.rand(M // block_size, K // block_size, dtype=torch.float32, device=device) * 0.1
scale_b = torch.rand(N // block_size, K // block_size, dtype=torch.float32, device=device) * 0.1

# 预热
for _ in range(10):
    _ = torch.matmul(A_bf16, B_bf16)
if device.type == 'cuda':
    torch.cuda.synchronize()

# 测试 bf16 GEMM
num_iterations = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(num_iterations):
    C_bf16 = torch.matmul(A_bf16, B_bf16)
end_event.record()
torch.cuda.synchronize()
bf16_latency = start_event.elapsed_time(end_event) / num_iterations

# 预热 int8
for _ in range(10):
    _ = torch._int_mm(A_int8, B_int8)
if device.type == 'cuda':
    torch.cuda.synchronize()

# 测试 int8 GEMM
start_event.record()
for _ in range(num_iterations):
    C_int8 = torch._int_mm(A_int8, B_int8)
end_event.record()
torch.cuda.synchronize()
int8_latency = start_event.elapsed_time(end_event) / num_iterations

# 预热 gemm_int8
for _ in range(10):
    _ = gemm_int8.matmul(A_int8, B_int8_t)
if device.type == 'cuda':
    torch.cuda.synchronize()

# 测试 gemm_int8 GEMM
start_event.record()
for _ in range(num_iterations):
    C_gemm_int8 = gemm_int8.matmul(A_int8, B_int8_t)
end_event.record()
torch.cuda.synchronize()
gemm_int8_latency = start_event.elapsed_time(end_event) / num_iterations

# 预热 gemm_int8 blockwise scaled
for _ in range(10):
    _ = gemm_int8.matmul_blockwise_scaled(A_int8, B_int8_t, scale_a, scale_b)
if device.type == 'cuda':
    torch.cuda.synchronize()

# 测试 gemm_int8 blockwise scaled GEMM
start_event.record()
for _ in range(num_iterations):
    C_gemm_int8_scaled = gemm_int8.matmul_blockwise_scaled(A_int8, B_int8_t, scale_a, scale_b)
end_event.record()
torch.cuda.synchronize()
gemm_int8_scaled_latency = start_event.elapsed_time(end_event) / num_iterations

# 计算 FLOPS (GEMM 的计算量: 2*M*N*K)
flops = 2 * M * N * K

# 计算 TFLOPS (FLOPS / (时间(秒) * 10^12))
bf16_tflops = flops / (bf16_latency * 1e-3) / 1e12
int8_tflops = flops / (int8_latency * 1e-3) / 1e12
gemm_int8_tflops = flops / (gemm_int8_latency * 1e-3) / 1e12
gemm_int8_scaled_tflops = flops / (gemm_int8_scaled_latency * 1e-3) / 1e12

# 输出结果
print(f"\n矩阵维度: M={M}, K={K}, N={N}")
print(f"总计算量: {flops / 1e9:.2f} GFLOPS")
print(f"迭代次数: {num_iterations}")

print(f"\n{'算子':<40} {'时延(ms)':<12} {'TFLOPS':<10}")
print("=" * 65)
print(f"{'bf16 GEMM (torch.matmul)':<40} {bf16_latency:<12.4f} {bf16_tflops:<10.2f}")
print(f"{'int8 GEMM (torch._int_mm)':<40} {int8_latency:<12.4f} {int8_tflops:<10.2f}")
print(f"{'int8 GEMM (gemm_int8.matmul)':<40} {gemm_int8_latency:<12.4f} {gemm_int8_tflops:<10.2f}")
print(f"{'int8 GEMM (gemm_int8.matmul_blockwise_scaled)':<40} {gemm_int8_scaled_latency:<12.4f} {gemm_int8_scaled_tflops:<10.2f}")

print(f"\n{'加速比对比':<50} {'倍数':<10}")
print("=" * 65)
print(f"{'bf16 vs torch._int_mm':<50} {bf16_latency / int8_latency:<10.2f}x")
print(f"{'bf16 vs gemm_int8':<50} {bf16_latency / gemm_int8_latency:<10.2f}x")
print(f"{'bf16 vs gemm_int8_scaled':<50} {bf16_latency / gemm_int8_scaled_latency:<10.2f}x")
print(f"{'torch._int_mm vs gemm_int8':<50} {int8_latency / gemm_int8_latency:<10.2f}x")
print(f"{'gemm_int8 vs gemm_int8_scaled':<50} {gemm_int8_latency / gemm_int8_scaled_latency:<10.2f}x")
