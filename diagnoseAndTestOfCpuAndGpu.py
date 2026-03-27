"""
GPU/CPU 性能诊断与测试工具
整合了以下功能：
- GPU 性能诊断
- 批量处理测试
- GPU vs CPU 详细对比
- 详细性能分析
"""
import torch
import time
import numpy as np
import os.path as osp
import glob

# 启用cuDNN自动优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def get_dataset_info():
    """获取数据集信息"""
    gallery_dir = osp.join('data', 'BallShow', 'bounding_box_test')
    query_dir = osp.join('data', 'BallShow', 'query')
    gallery_files = glob.glob(osp.join(gallery_dir, '*.jpg')) if osp.exists(gallery_dir) else []
    query_files = glob.glob(osp.join(query_dir, '*.jpg')) if osp.exists(query_dir) else []
    n = len(gallery_files) if gallery_files else 5000
    m = 768  # 特征维度
    return n, m, len(gallery_files), len(query_files)


def diagnose_gpu():
    """GPU 性能诊断"""
    print("=" * 60)
    print("GPU 性能诊断")
    print("=" * 60)

    print("\n1. CUDA状态:")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   设备数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n   ⚠️  警告: CUDA不可用,必须使用CPU,性能会很慢!")
        print("   解决方案:")
        print("   - 安装CUDA Toolkit")
        print("   - 安装PyTorch GPU版本")
        print("   - 检查NVIDIA驱动")
        return

    device = torch.device('cuda')

    # 预热
    print("\n2. 预热GPU...")
    dummy = torch.randn(100, 3, 256, 128).to(device)
    _ = dummy.sum()
    torch.cuda.synchronize()

    # 测试矩阵乘法速度
    print("\n3. 测试矩阵乘法速度:")

    n, m, actual_gallery, actual_query = get_dataset_info()

    print(f"   数据集规模:")
    print(f"   - Gallery 实际数量: {actual_gallery}")
    print(f"   - Query 实际数量: {actual_query}")
    print(f"   - 测试使用 Gallery: {n}")

    # 创建随机数据
    query = torch.randn(1, m).to(device)
    gallery = torch.randn(n, m).to(device)

    # GPU计算
    torch.cuda.synchronize()
    start = time.time()

    # 归一化
    query_norm = torch.norm(query, p=2, dim=1, keepdim=True)
    gallery_norm = torch.norm(gallery, p=2, dim=1, keepdim=True)
    query_normalized = query / (query_norm + 1e-8)
    gallery_normalized = gallery / (gallery_norm + 1e-8)

    # 计算相似度
    similarities = torch.mm(query_normalized, gallery_normalized.t())

    # Top-K
    k = 10
    top_values, top_indices = torch.topk(similarities, k=k, dim=1)

    torch.cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    print(f"   Gallery数量: {n}")
    print(f"   特征维度: {m}")
    print(f"   计算时间: {gpu_time:.2f} ms")
    print(f"   是否达标: {'OK' if gpu_time <= 30 else 'X'} (目标: ≤30ms)")

    # 纯矩阵乘法性能测试
    print("\n3.2 纯矩阵乘法性能测试:")
    query_rand = torch.randn(1, m).to(device)
    gallery_rand = torch.randn(n, m).to(device)

    torch.cuda.synchronize()
    start_mm = time.time()
    similarities_mm = torch.mm(query_rand, gallery_rand.t())
    top_values_mm, top_indices_mm = torch.topk(similarities_mm, k=k, dim=1)
    torch.cuda.synchronize()
    gpu_mm_time = (time.time() - start_mm) * 1000

    # CPU对比
    query_rand_cpu = query_rand.cpu().numpy()
    gallery_rand_cpu = gallery_rand.cpu().numpy()
    start_mm_cpu = time.time()
    similarities_cpu = np.dot(gallery_rand_cpu, query_rand_cpu.T).flatten()
    top_indices_cpu = np.argsort(similarities_cpu)[::-1][:k]
    cpu_mm_time = (time.time() - start_mm_cpu) * 1000

    print(f"   GPU矩阵乘法时间: {gpu_mm_time:.2f} ms")
    print(f"   CPU矩阵乘法时间: {cpu_mm_time:.2f} ms")
    if gpu_mm_time > 0:
        print(f"   GPU加速比: {cpu_mm_time/gpu_mm_time:.1f}x")
    print(f"   是否达标: {'OK' if gpu_mm_time <= 30 else 'X'} (目标: ≤30ms)")

    # 优化测试：半精度
    print("\n3.3 优化测试（半精度）:")
    with torch.cuda.amp.autocast(enabled=True):
        torch.cuda.synchronize()
        start_fp16 = time.time()

        query_norm_fp16 = torch.norm(query, p=2, dim=1, keepdim=True)
        gallery_norm_fp16 = torch.norm(gallery, p=2, dim=1, keepdim=True)
        query_normalized_fp16 = query / (query_norm_fp16 + 1e-8)
        gallery_normalized_fp16 = gallery / (gallery_norm_fp16 + 1e-8)
        similarities_fp16 = torch.mm(query_normalized_fp16, gallery_normalized_fp16.t())
        top_values_fp16, top_indices_fp16 = torch.topk(similarities_fp16, k=k, dim=1)

        torch.cuda.synchronize()
        gpu_time_fp16 = (time.time() - start_fp16) * 1000

    print(f"   FP16计算时间: {gpu_time_fp16:.2f} ms")
    print(f"   加速比: {gpu_time/gpu_time_fp16:.1f}x")

    # 检查Tensor Core支持
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        print("   GPU支持BF16 (Tensor Core)")
    elif torch.cuda.get_device_properties(0).major >= 7:
        print("   GPU支持FP16 Tensor Core (Volta+)")
    else:
        print("   GPU可能不支持Tensor Core")

    # 测试特征提取
    print("\n4. 测试特征提取速度:")
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2)
            self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=2)
            self.conv4 = torch.nn.Conv2d(256, 512, 3, stride=2)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))
            x = self.pool(x)
            return x.view(x.size(0), -1)

    model = DummyModel().to(device)
    model.eval()

    # 预热
    dummy_input = torch.randn(1, 3, 256, 128).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()

    # 测试
    times = []
    for _ in range(10):
        dummy_input = torch.randn(1, 3, 256, 128).to(device)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    print(f"   平均提取时间: {avg_time:.2f} ms (10次平均)")
    print(f"   是否达标: {'OK' if avg_time <= 40 else 'X'} (目标: ≤40ms)")

    # 显存使用
    print("\n5. 显存使用:")
    print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


def test_batch():
    """批量处理性能测试"""
    print("=" * 60)
    print("批量处理性能测试：GPU vs CPU")
    print("=" * 60)

    n, m, _, _ = get_dataset_info()
    print(f"\n数据集规模: Gallery={n}, 特征维度={m}")

    # 预计算Gallery
    gallery = torch.randn(n, m)
    gallery_normalized_cpu = gallery / np.linalg.norm(gallery.numpy(), axis=1, keepdims=True)
    gallery_normalized_gpu = torch.nn.functional.normalize(torch.randn(n, m).cuda(), p=2, dim=1)

    device = torch.device('cuda')

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]

    print("\n" + "=" * 60)
    print("不同Batch Size下的性能对比")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'GPU Time (ms)':<15} {'CPU Time (ms)':<15} {'加速比':<10} {'GPU更快?'}")
    print("-" * 70)

    results = []

    for batch_size in batch_sizes:
        query = torch.randn(batch_size, m)

        # GPU测试
        query_gpu = query.cuda()
        torch.cuda.synchronize()
        start_gpu = time.time()
        query_norm_gpu = torch.nn.functional.normalize(query_gpu, p=2, dim=1)
        similarities_gpu = torch.mm(query_norm_gpu, gallery_normalized_gpu.t())
        _, _ = torch.topk(similarities_gpu, k=10, dim=1)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_gpu) * 1000

        # CPU测试
        query_cpu = query.numpy()
        start_cpu = time.time()
        query_norm_cpu = query_cpu / np.linalg.norm(query_cpu, axis=1, keepdims=True)
        similarities_cpu = np.dot(query_norm_cpu, gallery_normalized_cpu.T)
        top_indices_cpu = np.argsort(-similarities_cpu, axis=1)[:, :10]
        cpu_time = (time.time() - start_cpu) * 1000

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        gpu_faster = "OK" if speedup > 1 else "X"

        print(f"{batch_size:<12} {gpu_time:<15.2f} {cpu_time:<15.2f} {speedup:<10.1f}x {gpu_faster}")
        results.append((batch_size, gpu_time, cpu_time, speedup))

    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)

    gpu_faster_batches = [r for r in results if r[3] > 1]
    if gpu_faster_batches:
        first_fast = gpu_faster_batches[0]
        print(f"\nGPU 开始比 CPU 快的 Batch Size: {first_fast[0]}")
        print(f"  此时加速比: {first_fast[3]:.1f}x")

    best = max(results, key=lambda x: x[3])
    print(f"\n最佳性能 (Batch Size={best[0]}):")
    print(f"  GPU: {best[1]:.2f} ms")
    print(f"  CPU: {best[2]:.2f} ms")
    print(f"  加速比: {best[3]:.1f}x")
    print(f"  单个Query平均时间: {best[1]/best[0]:.2f} ms")


def test_cpu_gpu():
    """GPU vs CPU 详细对比"""
    print("=" * 60)
    print("GPU vs CPU 详细对比")
    print("=" * 60)

    n, m, _, _ = get_dataset_info()
    print(f"\n数据集规模: Gallery={n}, 特征维度={m}")

    query = torch.randn(1, m)
    gallery = torch.randn(n, m)

    print("\n" + "=" * 60)
    print("1. 纯矩阵乘法对比")
    print("=" * 60)

    device = torch.device('cuda')
    query_gpu = query.to(device)
    gallery_gpu = gallery.to(device)

    # 预热
    dummy = torch.randn(100, m).to(device)
    _ = dummy.sum()
    torch.cuda.synchronize()

    # GPU 测试
    torch.cuda.synchronize()
    start = time.time()
    result_gpu = torch.mm(query_gpu, gallery_gpu.t())
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) * 1000
    print(f"GPU 矩阵乘法: {gpu_time:.4f} ms")

    # CPU 测试
    start = time.time()
    result_cpu = np.dot(gallery.numpy(), query.numpy().T)
    cpu_time = (time.time() - start) * 1000
    print(f"CPU 矩阵乘法 (NumPy): {cpu_time:.4f} ms")

    if gpu_time > 0:
        print(f"加速比: {cpu_time/gpu_time:.1f}x")

    print("\n" + "=" * 60)
    print("2. 完整流程对比（归一化 + 矩阵乘法 + Top-K）")
    print("=" * 60)

    # GPU 完整流程
    torch.cuda.synchronize()
    start = time.time()

    query_norm = torch.norm(query_gpu, p=2, dim=1, keepdim=True)
    gallery_norm = torch.norm(gallery_gpu, p=2, dim=1, keepdim=True)
    query_normalized = query_gpu / (query_norm + 1e-8)
    gallery_normalized = gallery_gpu / (gallery_norm + 1e-8)
    similarities = torch.mm(query_normalized, gallery_normalized.t())
    top_values, top_indices = torch.topk(similarities, k=10, dim=1)

    torch.cuda.synchronize()
    gpu_full_time = (time.time() - start) * 1000

    # CPU 完整流程
    start = time.time()

    query_cpu = query.numpy()
    gallery_cpu = gallery.numpy()
    query_norm_cpu = np.linalg.norm(query_cpu)
    gallery_norm_cpu = np.linalg.norm(gallery_cpu, axis=1)
    similarities_cpu = np.dot(gallery_cpu, query_cpu.T).flatten() / (query_norm_cpu * gallery_norm_cpu + 1e-8)
    top_indices_cpu = np.argsort(similarities_cpu)[::-1][:10]

    cpu_full_time = (time.time() - start) * 1000

    print(f"GPU 完整流程: {gpu_full_time:.2f} ms")
    print(f"CPU 完整流程: {cpu_full_time:.2f} ms")
    if gpu_full_time > 0:
        print(f"加速比: {cpu_full_time/gpu_full_time:.1f}x")

    print("\n" + "=" * 60)
    print("3. 为什么GPU反而慢？")
    print("=" * 60)
    print("""
原因分析：

1. 数据传输开销
   - 数据从CPU传输到GPU需要时间
   - 小批量数据（单个query）传输开销占比高

2. Kernel启动开销
   - 每个GPU操作都需要启动Kernel
   - 小操作的开销可能超过计算本身

3. 归一化操作不适合GPU
   - torch.norm 是逐元素操作
   - GPU优势在于大规模并行计算，小规模效率低

4. NumPy的优化
   - NumPy使用Intel MKL，针对小规模矩阵优化很好
   - CPU的缓存命中率更高

5. Top-K操作
   - GPU上的排序操作通常比CPU慢
   - 小数据量时，CPU的排序更快

结论：GPU 适合大规模、批量的计算任务
      单个query的小规模计算，CPU反而更快！
""")


def test_detailed():
    """详细性能分析"""
    print("=" * 60)
    print("详细性能分析：检索优化对比")
    print("=" * 60)

    device = torch.device('cuda')

    n, m, _, _ = get_dataset_info()
    print(f"\n数据集规模: Gallery={n}, 特征维度={m}")

    batch_size = 32
    query = torch.randn(batch_size, m).to(device)
    gallery = torch.randn(n, m).to(device)

    # 预热
    dummy = torch.randn(100, 3, 256, 128).to(device)
    _ = dummy.sum()
    torch.cuda.synchronize()

    iterations = 100

    print("\n【原始方案】每次查询重新归一化Gallery:")
    print("-" * 60)
    # 模拟原始方案：每次查询都归一化Gallery
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        # Query归一化
        q_norm = torch.norm(query, p=2, dim=1, keepdim=True)
        q_normed = query / (q_norm + 1e-8)
        # Gallery归一化（每次查询都重算！）
        g_norm = torch.norm(gallery, p=2, dim=1, keepdim=True)
        g_normed = gallery / (g_norm + 1e-8)
        # 矩阵乘法
        sim = torch.mm(q_normed, g_normed.t())
    torch.cuda.synchronize()
    time_old = (time.time() - start) * 1000 / iterations
    print(f"Query归一化+Gallery归一化+矩阵乘: {time_old:.2f} ms")

    print("\n【优化方案】Gallery预计算 + functional.normalize:")
    print("-" * 60)
    # 优化方案：Gallery只归一化一次
    gallery_normed = torch.nn.functional.normalize(gallery, p=2, dim=1)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        q_normed = torch.nn.functional.normalize(query, p=2, dim=1)
        sim = torch.mm(q_normed, gallery_normed.t())
    torch.cuda.synchronize()
    time_new = (time.time() - start) * 1000 / iterations
    print(f"Query归一化+矩阵乘: {time_new:.2f} ms")

    print("\n" + "=" * 60)
    print("优化效果总结")
    print("=" * 60)
    print(f"原始方案（每次重算Gallery）: {time_old:.2f} ms")
    print(f"优化后（Gallery预计算）:    {time_new:.2f} ms")
    print(f"加速比: {time_old/time_new:.2f}x")
    print(f"性能提升: {(time_old-time_new)/time_old*100:.1f}%")


def main():
    """运行所有测试"""
    diagnose_gpu()
    print("\n")
    test_batch()
    print("\n")
    test_cpu_gpu()
    print("\n")
    test_detailed()


if __name__ == '__main__':
    main()