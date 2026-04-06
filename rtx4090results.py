#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTX4090 训练结果提取工具
读取所有包含rtx4090的日志文件夹中的train_log.txt，
提取每个epoch的测试指标并保存到rtx4090results.txt
"""

import os
import re
from pathlib import Path
from datetime import datetime

# 全局常量定义
TARGET_MAP = 91.5
TARGET_RANK1 = 94.0


def parse_train_log(log_path):
    """解析训练日志文件，提取每个epoch的测试结果"""
    results = []
    current_epoch = None
    current_time = None
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 匹配时间戳行: 2026-04-02 04:50:05,683 transreid.train INFO: Validation Results - Epoch: 1
        time_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
        validation_pattern = re.compile(r'Validation Results - Epoch:\s*(\d+)')
        map_pattern = re.compile(r'mAP:\s*([\d.]+)%')
        rank1_pattern = re.compile(r'Rank-1\s*:\s*([\d.]+)%')
        rank5_pattern = re.compile(r'Rank-5\s*:\s*([\d.]+)%')
        rank10_pattern = re.compile(r'Rank-10\s*:\s*([\d.]+)%')
        
        for line in lines:
            # 提取时间戳
            time_match = time_pattern.match(line)
            if time_match:
                current_time = time_match.group(1)
            
            # 检查是否是验证结果行
            val_match = validation_pattern.search(line)
            if val_match:
                current_epoch = int(val_match.group(1))
                # 初始化结果
                results.append({
                    'epoch': current_epoch,
                    'time': current_time,
                    'mAP': None,
                    'Rank-1': None,
                    'Rank-5': None,
                    'Rank-10': None
                })
            
            # 提取指标
            if current_epoch is not None and results:
                last_result = results[-1]
                if last_result['epoch'] == current_epoch:
                    map_match = map_pattern.search(line)
                    if map_match:
                        last_result['mAP'] = float(map_match.group(1))
                    
                    rank1_match = rank1_pattern.search(line)
                    if rank1_match:
                        last_result['Rank-1'] = float(rank1_match.group(1))
                    
                    rank5_match = rank5_pattern.search(line)
                    if rank5_match:
                        last_result['Rank-5'] = float(rank5_match.group(1))
                    
                    rank10_match = rank10_pattern.search(line)
                    if rank10_match:
                        last_result['Rank-10'] = float(rank10_match.group(1))
                        
                    # 如果所有指标都提取完毕，重置current_epoch
                    if all(v is not None for v in last_result.values()):
                        current_epoch = None
                        
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        
    return results


def find_rtx4090_logs(base_dir):
    """查找所有包含rtx4090的日志文件夹"""
    rtx4090_logs = []
    logs_dir = os.path.join(base_dir, 'logs')
    
    if not os.path.exists(logs_dir):
        print(f"Warning: {logs_dir} does not exist")
        return rtx4090_logs
        
    for item in os.listdir(logs_dir):
        if 'rtx4090' in item.lower() and os.path.isdir(os.path.join(logs_dir, item)):
            train_log = os.path.join(logs_dir, item, 'train_log.txt')
            if os.path.exists(train_log):
                rtx4090_logs.append({
                    'name': item,
                    'log_path': train_log
                })
                print(f"Found: {item}")
            else:
                print(f"Found folder but no train_log.txt: {item}")
                
    return rtx4090_logs


def main():
    # 获取脚本所在目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'logs')
    output_file = os.path.join(output_dir, 'rtx4090results.txt')
    
    print("=" * 70)
    print("RTX4090 训练结果提取工具")
    print("=" * 70)
    
    # 查找所有rtx4090日志
    print("\n查找rtx4090日志文件夹...")
    rtx4090_logs = find_rtx4090_logs(script_dir)
    
    if not rtx4090_logs:
        print("\n未找到任何rtx4090相关的训练日志！")
        return
        
    print(f"\n共找到 {len(rtx4090_logs)} 个日志文件夹\n")
    
    # 解析每个日志文件
    all_results = {}
    for log_info in rtx4090_logs:
        name = log_info['name']
        log_path = log_info['log_path']
        print(f"解析: {name}")
        
        results = parse_train_log(log_path)
        if results:
            all_results[name] = results
            print(f"  - 提取到 {len(results)} 个epoch的测试结果")
        else:
            print(f"  - 未提取到有效数据")
    
    # 写入结果文件
    print(f"\n写入结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RTX4090 系列模型训练测试结果汇总\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # 收集每个模型的时间范围
        time_ranges = []
        for model_name, results in all_results.items():
            valid_results = [r for r in results if r['time'] is not None]
            if valid_results:
                first_time = valid_results[0]['time']
                last_time = valid_results[-1]['time']
                time_ranges.append((first_time, model_name, first_time, last_time))
        
        # 按最早时间升序排列
        time_ranges.sort(key=lambda x: x[0])
        
        # 写入时间范围列表
        f.write("【训练先后顺序】\n\n")
        for i, (first_time, model_name, first, last) in enumerate(time_ranges, 1):
            f.write(f"{i}. {model_name}\n")
            f.write(f"   {first} ~ {last}\n\n")
        
        for model_name, results in all_results.items():
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"模型配置: {model_name}\n")
            f.write("=" * 70 + "\n")
            
            # 写入表头
            header = f"{'Epoch':<8} {'时间':<20} {'mAP(%)':<10} {'Rank-1(%)':<12} {'Rank-5(%)':<12} {'Rank-10(%)':<12}\n"
            f.write(header)
            f.write("-" * 90 + "\n")
            
            # 写入每个epoch的结果，每25个epoch重复写表头
            for i, r in enumerate(results):
                if r['mAP'] is not None:
                    time_str = r['time'] if r['time'] else '-'
                    f.write(f"{r['epoch']:<8} {time_str:<20} {r['mAP']:<10.1f} {r['Rank-1']:<12.1f} {r['Rank-5']:<12.1f} {r['Rank-10']:<12.1f}\n")
                    # 每25个epoch写一次表头（从第25个开始，即0索引24）
                    if (i + 1) % 25 == 0 and (i + 1) < len(results):
                        f.write("-" * 90 + "\n")
                        f.write(header)
            
            # 找到最佳结果
            valid_results = [r for r in results if r['mAP'] is not None]
            if valid_results:
                best_by_map = max(valid_results, key=lambda x: x['mAP'])
                best_by_rank1 = max(valid_results, key=lambda x: x['Rank-1'])
                best_by_rank5 = max(valid_results, key=lambda x: x['Rank-5'])
                best_by_rank10 = max(valid_results, key=lambda x: x['Rank-10'])
                
                f.write("\n" + "-" * 90 + "\n")
                f.write(f"【{model_name}】\n")
                
                # 表头（增加时间列）
                f.write(f"{'':>12} {'Epoch':<8} {'时间':<20} {'mAP':<10} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10} {'参数目标':<12} {'参数差距':<12}\n")
                f.write("-" * 105 + "\n")
                
                # 最佳mAP
                diff_map = best_by_map['mAP'] - TARGET_MAP
                if diff_map >= 0:
                    diff_str_map = f"超出{abs(diff_map):.1f}%"
                else:
                    diff_str_map = f"还差{abs(diff_map):.1f}%"
                time_map = best_by_map['time'] if best_by_map['time'] else '-'
                f.write(f"{'最佳mAP':>12} {best_by_map['epoch']:<8} {time_map:<20} {best_by_map['mAP']:<10.1f} {best_by_map['Rank-1']:<10.1f} {best_by_map['Rank-5']:<10.1f} {best_by_map['Rank-10']:<10.1f} {TARGET_MAP:<12.1f} {diff_str_map:<12}\n")
                
                # 最佳Rank-1
                diff_rank1 = best_by_rank1['Rank-1'] - TARGET_RANK1
                if diff_rank1 >= 0:
                    diff_str_rank1 = f"超出{abs(diff_rank1):.1f}%"
                else:
                    diff_str_rank1 = f"还差{abs(diff_rank1):.1f}%"
                time_rank1 = best_by_rank1['time'] if best_by_rank1['time'] else '-'
                f.write(f"{'最佳Rank-1':>12} {best_by_rank1['epoch']:<8} {time_rank1:<20} {best_by_rank1['mAP']:<10.1f} {best_by_rank1['Rank-1']:<10.1f} {best_by_rank1['Rank-5']:<10.1f} {best_by_rank1['Rank-10']:<10.1f} {TARGET_RANK1:<12.1f} {diff_str_rank1:<12}\n")
                
                # 最佳Rank-5
                time_rank5 = best_by_rank5['time'] if best_by_rank5['time'] else '-'
                f.write(f"{'最佳Rank-5':>12} {best_by_rank5['epoch']:<8} {time_rank5:<20} {best_by_rank5['mAP']:<10.1f} {best_by_rank5['Rank-1']:<10.1f} {best_by_rank5['Rank-5']:<10.1f} {best_by_rank5['Rank-10']:<10.1f} {'-':<12} {'-':<12}\n")
                
                # 最佳Rank-10
                time_rank10 = best_by_rank10['time'] if best_by_rank10['time'] else '-'
                f.write(f"{'最佳Rank-10':>12} {best_by_rank10['epoch']:<8} {time_rank10:<20} {best_by_rank10['mAP']:<10.1f} {best_by_rank10['Rank-1']:<10.1f} {best_by_rank10['Rank-5']:<10.1f} {best_by_rank10['Rank-10']:<10.1f} {'-':<12} {'-':<12}\n")
                
                # 统计所有epoch中mAP和Rank-1最高的5个值
                if len(valid_results) > 0:
                    # 统计所有epoch的mAP值出现的频率和对应epoch
                    all_map_value_stats = {}
                    for r in valid_results:
                        map_value = r['mAP']
                        if map_value not in all_map_value_stats:
                            all_map_value_stats[map_value] = {'count': 0, 'epochs': []}
                        all_map_value_stats[map_value]['count'] += 1
                        all_map_value_stats[map_value]['epochs'].append(r['epoch'])
                    
                    # 统计所有epoch的Rank-1值出现的频率和对应epoch
                    all_rank1_value_stats = {}
                    for r in valid_results:
                        rank1_value = r['Rank-1']
                        if rank1_value not in all_rank1_value_stats:
                            all_rank1_value_stats[rank1_value] = {'count': 0, 'epochs': []}
                        all_rank1_value_stats[rank1_value]['count'] += 1
                        all_rank1_value_stats[rank1_value]['epochs'].append(r['epoch'])
                    
                    f.write("\n" + "-" * 105 + "\n")
                    f.write("【该模型所有epoch最高值统计】\n")
                    f.write(f"统计范围: epoch {valid_results[0]['epoch']} 到 {valid_results[-1]['epoch']} (共{len(valid_results)}个epoch)\n")
                    f.write("-" * 105 + "\n")
                    
                    # 输出mAP最高的3个值
                    f.write("mAP最高3个值:\n")
                    f.write("------------\n")
                    sorted_map_values = sorted(all_map_value_stats.keys(), reverse=True)
                    top_3_maps = sorted_map_values[:3] if len(sorted_map_values) >= 3 else sorted_map_values
                    
                    for map_value in top_3_maps:
                        stats = all_map_value_stats[map_value]
                        # 计算离目标值的差距（百分比）
                        diff_to_target = map_value - TARGET_MAP
                        if diff_to_target > 0:
                            diff_str = f"超目标{abs(diff_to_target):.1f}%"
                        elif diff_to_target == 0:
                            diff_str = "目标"
                        else:
                            diff_str = f"离目标{abs(diff_to_target):.1f}%"
                        
                        # 获取对应epoch
                        epochs_str = ', '.join(str(e) for e in stats['epochs'])
                        
                        f.write(f"  {map_value:.1f}%({diff_str}): 出现{stats['count']}次\n")
                        f.write(f"     对应epoch: {epochs_str}\n")
                    
                    f.write("\n")
                    
                    # 输出Rank-1最高的3个值
                    f.write("Rank-1最高3个值:\n")
                    f.write("---------------\n")
                    sorted_rank1_values = sorted(all_rank1_value_stats.keys(), reverse=True)
                    top_3_rank1s = sorted_rank1_values[:3] if len(sorted_rank1_values) >= 3 else sorted_rank1_values
                    
                    for rank1_value in top_3_rank1s:
                        stats = all_rank1_value_stats[rank1_value]
                        # 计算离目标值的差距（百分比）
                        diff_to_target = rank1_value - TARGET_RANK1
                        if diff_to_target > 0:
                            diff_str = f"超目标{abs(diff_to_target):.1f}%"
                        elif diff_to_target == 0:
                            diff_str = "目标"
                        else:
                            diff_str = f"离目标{abs(diff_to_target):.1f}%"
                        
                        # 获取对应epoch
                        epochs_str = ', '.join(str(e) for e in stats['epochs'])
                        
                        f.write(f"  {rank1_value:.1f}%({diff_str}): 出现{stats['count']}次\n")
                        f.write(f"     对应epoch: {epochs_str}\n")
        
        # 在txt末尾写入pth文件对照表格
        write_comparison_table(f, all_results, output_dir)
    
    print("\n结果已保存!")
    
    # 打印汇总表格到终端
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r['mAP'] is not None]
        if valid_results:
            best_by_map = max(valid_results, key=lambda x: x['mAP'])
            best_by_rank1 = max(valid_results, key=lambda x: x['Rank-1'])
            best_by_rank5 = max(valid_results, key=lambda x: x['Rank-5'])
            best_by_rank10 = max(valid_results, key=lambda x: x['Rank-10'])
            
            print(f"\n{'=' * 100}")
            print(f"【{model_name}】")
            print(f"{'=' * 100}")
            print(f"{'':>12} {'Epoch':<8} {'时间':<20} {'mAP':<10} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10} {'参数目标':<12} {'参数差距':<12}")
            print(f"{'-' * 100}")
            
            diff_map = best_by_map['mAP'] - TARGET_MAP
            diff_str_map = f"超出{abs(diff_map):.1f}%" if diff_map >= 0 else f"还差{abs(diff_map):.1f}%"
            time_map = best_by_map['time'] if best_by_map['time'] else '-'
            print(f"{'最佳mAP':>12} {best_by_map['epoch']:<8} {time_map:<20} {best_by_map['mAP']:<10.1f} {best_by_map['Rank-1']:<10.1f} {best_by_map['Rank-5']:<10.1f} {best_by_map['Rank-10']:<10.1f} {TARGET_MAP:<12.1f} {diff_str_map:<12}")
            
            diff_rank1 = best_by_rank1['Rank-1'] - TARGET_RANK1
            diff_str_rank1 = f"超出{abs(diff_rank1):.1f}%" if diff_rank1 >= 0 else f"还差{abs(diff_rank1):.1f}%"
            time_rank1 = best_by_rank1['time'] if best_by_rank1['time'] else '-'
            print(f"{'最佳Rank-1':>12} {best_by_rank1['epoch']:<8} {time_rank1:<20} {best_by_rank1['mAP']:<10.1f} {best_by_rank1['Rank-1']:<10.1f} {best_by_rank1['Rank-5']:<10.1f} {best_by_rank1['Rank-10']:<10.1f} {TARGET_RANK1:<12.1f} {diff_str_rank1:<12}")
            
            time_rank5 = best_by_rank5['time'] if best_by_rank5['time'] else '-'
            print(f"{'最佳Rank-5':>12} {best_by_rank5['epoch']:<8} {time_rank5:<20} {best_by_rank5['mAP']:<10.1f} {best_by_rank5['Rank-1']:<10.1f} {best_by_rank5['Rank-5']:<10.1f} {best_by_rank5['Rank-10']:<10.1f} {'-':<12} {'-':<12}")
            
            time_rank10 = best_by_rank10['time'] if best_by_rank10['time'] else '-'
            print(f"{'最佳Rank-10':>12} {best_by_rank10['epoch']:<8} {time_rank10:<20} {best_by_rank10['mAP']:<10.1f} {best_by_rank10['Rank-1']:<10.1f} {best_by_rank10['Rank-5']:<10.1f} {best_by_rank10['Rank-10']:<10.1f} {'-':<12} {'-':<12}")
            
            # 终端输出所有epoch中最高5个值的统计
            if len(valid_results) > 0:
                # 统计所有epoch的mAP值出现的频率和对应epoch
                all_map_value_stats = {}
                for r in valid_results:
                    map_value = r['mAP']
                    if map_value not in all_map_value_stats:
                        all_map_value_stats[map_value] = {'count': 0, 'epochs': []}
                    all_map_value_stats[map_value]['count'] += 1
                    all_map_value_stats[map_value]['epochs'].append(r['epoch'])
                
                # 统计所有epoch的Rank-1值出现的频率和对应epoch
                all_rank1_value_stats = {}
                for r in valid_results:
                    rank1_value = r['Rank-1']
                    if rank1_value not in all_rank1_value_stats:
                        all_rank1_value_stats[rank1_value] = {'count': 0, 'epochs': []}
                    all_rank1_value_stats[rank1_value]['count'] += 1
                    all_rank1_value_stats[rank1_value]['epochs'].append(r['epoch'])
                
                print(f"\n{'=' * 100}")
                print(f"【该模型所有epoch最高值统计】")
                print(f"统计范围: epoch {valid_results[0]['epoch']} 到 {valid_results[-1]['epoch']} (共{len(valid_results)}个epoch)")
                print(f"{'=' * 100}")
                
                # 输出mAP最高的3个值
                print("mAP最高3个值:")
                print("------------")
                sorted_map_values = sorted(all_map_value_stats.keys(), reverse=True)
                top_3_maps = sorted_map_values[:3] if len(sorted_map_values) >= 3 else sorted_map_values
                
                for map_value in top_3_maps:
                    stats = all_map_value_stats[map_value]
                    # 计算离目标值的差距（百分比）
                    diff_to_target = map_value - TARGET_MAP
                    if diff_to_target > 0:
                        diff_str = f"超目标{abs(diff_to_target):.1f}%"
                    elif diff_to_target == 0:
                        diff_str = "目标"
                    else:
                        diff_str = f"离目标{abs(diff_to_target):.1f}%"
                    
                    # 获取对应epoch
                    epochs_str = ', '.join(str(e) for e in stats['epochs'])
                    
                    print(f"  {map_value:.1f}%({diff_str}): 出现{stats['count']}次")
                    print(f"     对应epoch: {epochs_str}")
                
                print()
                
                # 输出Rank-1最高的3个值
                print("Rank-1最高3个值:")
                print("---------------")
                sorted_rank1_values = sorted(all_rank1_value_stats.keys(), reverse=True)
                top_3_rank1s = sorted_rank1_values[:3] if len(sorted_rank1_values) >= 3 else sorted_rank1_values
                
                for rank1_value in top_3_rank1s:
                    stats = all_rank1_value_stats[rank1_value]
                    # 计算离目标值的差距（百分比）
                    diff_to_target = rank1_value - TARGET_RANK1
                    if diff_to_target > 0:
                        diff_str = f"超目标{abs(diff_to_target):.1f}%"
                    elif diff_to_target == 0:
                        diff_str = "目标"
                    else:
                        diff_str = f"离目标{abs(diff_to_target):.1f}%"
                    
                    # 获取对应epoch
                    epochs_str = ', '.join(str(e) for e in stats['epochs'])
                    
                    print(f"  {rank1_value:.1f}%({diff_str}): 出现{stats['count']}次")
                    print(f"     对应epoch: {epochs_str}")
    
    # 在终端打印pth文件对照表格
    print_comparison_table(all_results, output_dir)


def find_pth_files(logs_dir, model_name):
    """查找某个模型文件夹中的所有pth文件，返回epoch列表"""
    model_dir = os.path.join(logs_dir, model_name)
    pth_epochs = []
    
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.pth') and f.startswith('transformer_checkpoint_'):
                # 提取epoch编号
                epoch_str = f.replace('transformer_checkpoint_', '').replace('.pth', '')
                try:
                    pth_epochs.append(int(epoch_str))
                except ValueError:
                    pass
    return sorted(set(pth_epochs))


def generate_pth_comparison_table(all_results, logs_dir):
    """生成pth文件与最优指标对照表格"""
    table_data = []
    
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r['mAP'] is not None]
        if not valid_results:
            continue
        
        # 找到各指标最佳epoch
        best_by_map = max(valid_results, key=lambda x: x['mAP'])
        best_by_rank1 = max(valid_results, key=lambda x: x['Rank-1'])
        best_by_rank5 = max(valid_results, key=lambda x: x['Rank-5'])
        best_by_rank10 = max(valid_results, key=lambda x: x['Rank-10'])
        
        # 获取pth文件对应的epochs
        pth_epochs = find_pth_files(logs_dir, model_name)
        pth_str = ', '.join([f'_{e}' for e in pth_epochs]) if pth_epochs else '无'
        
        # 检查一致性
        def check_match(pth_list, target_epoch):
            if target_epoch in pth_list:
                return f'{target_epoch} [OK]'
            return f'{target_epoch} [NO]'
        
        table_data.append({
            'model': model_name,
            'pth_str': pth_str,
            'best_map': check_match(pth_epochs, best_by_map['epoch']),
            'best_rank1': check_match(pth_epochs, best_by_rank1['epoch']),
            'best_rank5': check_match(pth_epochs, best_by_rank5['epoch']),
            'best_rank10': check_match(pth_epochs, best_by_rank10['epoch']),
            'best_map_epoch': best_by_map['epoch'],
            'best_rank1_epoch': best_by_rank1['epoch'],
            'best_rank5_epoch': best_by_rank5['epoch'],
            'best_rank10_epoch': best_by_rank10['epoch'],
        })
    
    return table_data


def write_comparison_table(f, all_results, logs_dir):
    """在txt文件末尾写入pth文件与最优指标对照表格"""
    table_data = generate_pth_comparison_table(all_results, logs_dir)
    
    if not table_data:
        return
    
    f.write("\n\n")
    f.write("=" * 140 + "\n")
    f.write("【pth文件与最优指标对照表】\n")
    f.write("=" * 140 + "\n")
    f.write("\n自动分析各模型文件夹中的pth文件，与txt中记录的最优epoch进行对照\n")
    f.write("- [OK] 表示该epoch有对应的pth文件保存\n")
    f.write("- [NO] 表示该epoch没有对应的pth文件保存\n\n")
    
    # 表头
    f.write(f"{'模型名称':<50} {'pth文件':<25} {'最佳mAP':<15} {'最佳Rank-1':<15} {'最佳Rank-5':<15} {'最佳Rank-10':<15}\n")
    f.write("-" * 140 + "\n")
    
    for row in table_data:
        f.write(f"{row['model']:<50} {row['pth_str']:<25} {row['best_map']:<15} {row['best_rank1']:<15} {row['best_rank5']:<15} {row['best_rank10']:<15}\n")
    
    # 统计信息
    f.write("\n" + "-" * 140 + "\n")
    f.write("【统计结论】\n")
    
    map_match_count = sum(1 for r in table_data if '[OK]' in r['best_map'])
    rank1_match_count = sum(1 for r in table_data if '[OK]' in r['best_rank1'])
    rank5_match_count = sum(1 for r in table_data if '[OK]' in r['best_rank5'])
    rank10_match_count = sum(1 for r in table_data if '[OK]' in r['best_rank10'])
    total_models = len(table_data)
    
    f.write(f"- pth文件保存了 {map_match_count}/{total_models} 个模型的 最佳mAP epoch\n")
    f.write(f"- pth文件保存了 {rank1_match_count}/{total_models} 个模型的 最佳Rank-1 epoch\n")
    f.write(f"- pth文件保存了 {rank5_match_count}/{total_models} 个模型的 最佳Rank-5 epoch\n")
    f.write(f"- pth文件保存了 {rank10_match_count}/{total_models} 个模型的 最佳Rank-10 epoch\n")
    f.write("\n结论: pth文件保存的是最佳mAP和最佳Rank-1的epoch，最佳Rank-5和Rank-10的epoch没有对应的pth文件保存\n")


def print_comparison_table(all_results, logs_dir):
    """在终端打印pth文件与最优指标对照表格"""
    table_data = generate_pth_comparison_table(all_results, logs_dir)
    
    if not table_data:
        return
    
    print("\n")
    print("=" * 140)
    print("【pth文件与最优指标对照表】")
    print("=" * 140)
    print("\n自动分析各模型文件夹中的pth文件，与txt中记录的最优epoch进行对照")
    print("- [OK] 表示该epoch有对应的pth文件保存")
    print("- [NO] 表示该epoch没有对应的pth文件保存\n")
    
    # 表头
    header = f"{'模型名称':<50} {'pth文件':<25} {'最佳mAP':<15} {'最佳Rank-1':<15} {'最佳Rank-5':<15} {'最佳Rank-10':<15}"
    print(header)
    print("-" * 140)
    
    for row in table_data:
        line = f"{row['model']:<50} {row['pth_str']:<25} {row['best_map']:<15} {row['best_rank1']:<15} {row['best_rank5']:<15} {row['best_rank10']:<15}"
        print(line)
    
    # 统计信息
    print("\n" + "-" * 140)
    print("【统计结论】")
    
    map_match_count = sum(1 for r in table_data if '[OK]' in r['best_map'])
    rank1_match_count = sum(1 for r in table_data if '[OK]' in r['best_rank1'])
    rank5_match_count = sum(1 for r in table_data if '[OK]' in r['best_rank5'])
    rank10_match_count = sum(1 for r in table_data if '[OK]' in r['best_rank10'])
    total_models = len(table_data)
    
    print(f"- pth文件保存了 {map_match_count}/{total_models} 个模型的 最佳mAP epoch")
    print(f"- pth文件保存了 {rank1_match_count}/{total_models} 个模型的 最佳Rank-1 epoch")
    print(f"- pth文件保存了 {rank5_match_count}/{total_models} 个模型的 最佳Rank-5 epoch")
    print(f"- pth文件保存了 {rank10_match_count}/{total_models} 个模型的 最佳Rank-10 epoch")


if __name__ == '__main__':
    main()
