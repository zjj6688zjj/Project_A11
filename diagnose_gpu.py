"""
快速诊断GPU性能问题
"""
import torch
import time
import numpy as np

print("="*60)
print("GPU性能诊断")
print("="*60)

# 1. 检查CUDA
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

# 2. 测试GPU性能
if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # 预热
    print("\n2. 预热GPU...")
    dummy = torch.randn(100, 3, 256, 128).to(device)
    _ = dummy.sum()
    torch.cuda.synchronize()
    
    # 测试矩阵乘法速度
    print("\n3. 测试矩阵乘法速度:")
    n = 5000  # gallery数量
    m = 768   # 特征维度
    
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
    print(f"   是否达标: {'✓' if gpu_time <= 30 else '✗'} (目标: ≤30ms)")
    
    # 测试CPU对比
    print("\n4. CPU vs GPU对比:")
    query_cpu = query.cpu().numpy()
    gallery_cpu = gallery.cpu().numpy()
    
    start = time.time()
    query_norm_cpu = np.linalg.norm(query_cpu)
    gallery_norm_cpu = np.linalg.norm(gallery_cpu, axis=1)
    similarities_cpu = np.dot(gallery_cpu, query_cpu.T).flatten() / (query_norm_cpu * gallery_norm_cpu)
    top_indices_cpu = np.argsort(similarities_cpu)[::-1][:k]
    cpu_time = (time.time() - start) * 1000
    
    print(f"   CPU计算时间: {cpu_time:.2f} ms")
    print(f"   GPU计算时间: {gpu_time:.2f} ms")
    print(f"   GPU加速比: {cpu_time/gpu_time:.1f}x")
    
    # 测试特征提取
    print("\n5. 测试特征提取速度:")
    
    # 模拟ViT模型特征提取
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2)
            self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=2)
            self.conv4 = torch.nn.Conv2d(256, 512, 3, stride=2)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.conv4(x)
            x = torch.relu(x)
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
    for i in range(10):
        dummy_input = torch.randn(1, 3, 256, 128).to(device)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"   平均提取时间: {avg_time:.2f} ms (10次平均)")
    print(f"   是否达标: {'✓' if avg_time <= 40 else '✗'} (目标: ≤40ms)")
    
    # 显存使用
    print("\n6. 显存使用:")
    print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print("\n" + "="*60)
print("诊断完成")
print("="*60)

# 问题诊断
print("\n问题诊断:")
if not torch.cuda.is_available():
    print("❌ 严重问题: CUDA不可用")
    print("   当前速度: CPU运行,特征提取~6秒,查询匹配~200秒")
    print("   目标速度: GPU运行,特征提取<40ms,查询匹配<30ms")
    print("\n   必须解决:")
    print("   1. 安装CUDA Toolkit (11.x或12.x)")
    print("   2. 重新安装PyTorch GPU版本:")
    print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
else:
    gpu_time = (time.time() - start) * 1000 if 'gpu_time' in locals() else 999
    if gpu_time <= 30:
        print("✓ GPU正常,性能达标")
    else:
        print("⚠️  GPU可用但性能未达标,可能原因:")
        print("   - 模型仍在CPU上运行")
        print("   - 没有使用批量处理")
        print("   - 数据传输开销大")
