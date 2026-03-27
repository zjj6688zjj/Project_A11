"""
提取所有测试日志中的性能指标
包括每次检测的 epoch 数、reranking 状态、测试时间
"""
import re
import os
from datetime import datetime

logs_dir = 'logs'

results = []

# 遍历 logs 下所有子目录
for log_dir_name in os.listdir(logs_dir):
    log_dir_path = os.path.join(logs_dir, log_dir_name)
    
    if not os.path.isdir(log_dir_path):
        continue
    
    test_log_path = os.path.join(log_dir_path, 'test_log.txt')
    
    if not os.path.exists(test_log_path):
        continue
    
    with open(test_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    config_name = log_dir_name.replace('BallShow_', '')
    
    reranking = None
    for line in lines:
        if 'RE_RANKING:' in line:
            if 'False' in line:
                reranking = False
            elif 'True' in line:
                reranking = True
            break
    
    # 解析测试结果 - 使用更可靠的方法
    epoch_results = {}
    
    for i, line in enumerate(lines):
        if 'Testing with weight:' in line:
            match = re.search(r'checkpoint_(\d+)\.pth', line)
            if not match:
                continue
            epoch = int(match.group(1))
            
            # 时间戳在上一行
            start_time = None
            if i > 0:
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[i-1])
                if time_match:
                    try:
                        start_time = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
                    except:
                        pass
            
            epoch_results[epoch] = {'epoch': epoch, 'start_time': start_time, 'start_line': i}
    
    # 解析性能指标
    for i, line in enumerate(lines):
        if 'mAP:' in line and 'CMC' not in line:
            match = re.search(r'mAP:\s*(\d+\.?\d*)%', line)
            if match:
                mAP = float(match.group(1))
                # 找到最近的 epoch
                for epoch in epoch_results:
                    if 'mAP' not in epoch_results[epoch] and epoch_results[epoch]['start_line'] < i:
                        epoch_results[epoch]['mAP'] = mAP
                        break
        
        if 'Rank-1' in line:
            match = re.search(r'Rank-1\s*:\s*(\d+\.?\d*)%', line)
            if match:
                r1 = float(match.group(1))
                for epoch in epoch_results:
                    if 'rank1' not in epoch_results[epoch] and epoch_results[epoch]['start_line'] < i:
                        epoch_results[epoch]['rank1'] = r1
                        break
        
        if 'Rank-10' in line:
            match = re.search(r'Rank-10\s*:\s*(\d+\.?\d*)%', line)
            if match:
                r10 = float(match.group(1))
                for epoch in epoch_results:
                    if 'rank10' not in epoch_results[epoch] and epoch_results[epoch]['start_line'] < i:
                        epoch_results[epoch]['rank10'] = r10
                        # 计算耗时 - 从当前行获取结束时间
                        end_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if end_match:
                            try:
                                end_time = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S')
                                if epoch_results[epoch]['start_time']:
                                    epoch_results[epoch]['duration'] = (end_time - epoch_results[epoch]['start_time']).total_seconds()
                            except:
                                pass
                        break
    
    # 过滤有效结果
    valid_results = [v for v in epoch_results.values() if 'mAP' in v and 'rank1' in v]
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['mAP'] if x['mAP'] else 0)
        
        results.append({
            'folder': log_dir_name,
            'config': config_name,
            'reranking': reranking,
            'all_results': valid_results,
            'best': best
        })

# 按 mAP 排序
results.sort(key=lambda x: x['best']['mAP'] if x['best']['mAP'] else 0, reverse=True)

# 输出汇总表
print("=" * 130)
print("测试结果汇总（按最佳 mAP 排序）")
print("=" * 130)
print(f"{'配置文件':<30} {'Reranking':<10} {'最佳Epoch':<10} {'mAP':<10} {'Rank-1':<10} {'Rank-10':<10} {'测试耗时(s)':<12}")
print("-" * 130)

for r in results:
    config_name = r['config']
    reranking_str = '开启' if r['reranking'] else ('关闭' if r['reranking'] is False else '未知')
    best = r['best']
    duration_str = f"{best['duration']:.0f}s" if 'duration' in best and best['duration'] else 'N/A'
    
    print(f"{config_name:<30} {reranking_str:<10} {best['epoch']:<10} {best['mAP']:.1f}%{'':<6} {best['rank1']:.1f}%{'':<6} {best['rank10']:.1f}%{'':<6} {duration_str:<12}")

print("\n" + "=" * 130)
print("详细测试记录")
print("=" * 130)

for r in results:
    config_name = r['config']
    reranking_str = '开启' if r['reranking'] else ('关闭' if r['reranking'] is False else '未知')
    
    print(f"\n[{config_name}] (Reranking: {reranking_str})")
    print("-" * 90)
    print(f"{'Epoch':<8} {'mAP':<10} {'Rank-1':<10} {'Rank-10':<10} {'耗时(s)':<12}")
    print("-" * 90)
    
    # 按 epoch 排序
    sorted_results = sorted(r['all_results'], key=lambda x: x['epoch'])
    
    for er in sorted_results:
        duration_str = f"{er['duration']:.0f}s" if 'duration' in er and er['duration'] else 'N/A'
        print(f"{er['epoch']:<8} {er['mAP']:.1f}%{'':<6} {er['rank1']:.1f}%{'':<6} {er['rank10']:.1f}%{'':<6} {duration_str:<12}")
    
    # 最佳结果标记
    best_epoch = r['best']['epoch']
    print(f"\n  最佳 Epoch: {best_epoch}, mAP: {r['best']['mAP']:.1f}%, Rank-1: {r['best']['rank1']:.1f}%")

print("\n" + "=" * 130)
print("最佳配置")
print("=" * 130)
if results:
    best = max(results, key=lambda x: x['best']['mAP'] if x['best']['mAP'] else 0)
    print(f"最佳配置: {best['config']}")
    print(f"  Reranking: {'开启' if best['reranking'] else '关闭'}")
    print(f"  最佳 Epoch: {best['best']['epoch']}")
    print(f"  mAP: {best['best']['mAP']:.1f}%")
    print(f"  Rank-1: {best['best']['rank1']:.1f}%")
    print(f"  Rank-10: {best['best']['rank10']:.1f}%")
    if 'duration' in best['best'] and best['best']['duration']:
        print(f"  测试耗时: {best['best']['duration']:.0f}s")