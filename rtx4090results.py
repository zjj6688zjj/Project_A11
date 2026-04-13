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
    """解析训练日志文件，提取每个epoch的测试结果和学习率"""
    results = []
    current_epoch = None
    current_time = None
    current_lr = None
    epoch_lr_values = {}  # 用于记录每个epoch中所有batch的学习率
    
    # 用于检测断点：存储 (resume_epoch, end_epoch, end_time)
    # 表示该次训练从 resume_epoch 开始，上一次训练到 end_epoch 结束
    breaks = []
    
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
        # 检测学习率: Base Lr: 2.50e-03
        lr_pattern = re.compile(r'Base Lr:\s*([\d.eE+-]+)')
        # 检测断点续训的标记: start training from epoch 128
        resume_pattern = re.compile(r'start training from epoch\s*(\d+)')
        
        # 新增：记录当前resume_epoch，用于标记该session的起始时间
        current_resume_epoch_for_session = None
        # 新增：标记是否需要查找该session的起始时间
        waiting_for_session_start = False
        # 新增：记录前一个session结束时results的索引（用于确定段的结束epoch）
        prev_session_end_index = 0
        
        for line in lines:
            # 检测断点续训标记: "start training from epoch X"
            # 【修复】只有当 resume_epoch > 1 或者前面确实存在训练记录时，才记录为断点
            resume_match = resume_pattern.search(line)
            if resume_match:
                resume_epoch = int(resume_match.group(1))
                
                # 检查是否已经记录过同一个 resume_epoch（去重）
                already_recorded = any(b[0] == resume_epoch for b in breaks)
                if not already_recorded:
                    # 【关键修复】前一个session的结束 = 当前断点发生前results的最后一条
                    prev_session_end_index = len(results) - 1
                    
                    # 从已经解析的results中找 resume_epoch - 1 的时间（作为前一个session结束时间）
                    last_time_before_break = None
                    if resume_epoch > 1:
                        for r in results:
                            if r['epoch'] == resume_epoch - 1 and r['time'] is not None:
                                last_time_before_break = r['time']
                                break
                    
                    # breaks 格式: (resume_epoch, last_time_before_break, prev_session_end_index, is_epoch1_restart)
                    # is_epoch1_restart = True 表示这是从 checkpoint 恢复后重新从 epoch 1 开始
                    is_epoch1_restart = (resume_epoch == 1)
                    
                    # 【关键修复】只有当前面确实存在训练记录时才记录为断点
                    # 如果 results 为空，说明这是首次训练开始，不是断点
                    if len(results) > 0:
                        breaks.append((resume_epoch, last_time_before_break, prev_session_end_index, is_epoch1_restart))
                
                # 【关键修复】记录当前resume_epoch，用于判断后续的validation结果属于哪个session
                current_resume_epoch_for_session = resume_epoch
                waiting_for_session_start = True
                # 跳过这行，不更新current_time
                continue
            
            # 提取时间戳
            time_match = time_pattern.match(line)
            
            # 检查是否是验证结果行
            val_match = validation_pattern.search(line)
            if val_match:
                current_epoch = int(val_match.group(1))
                current_time = time_match.group(1) if time_match else None
                
                # 【关键修复】检查是否是新session的第一个验证结果
                is_new_session_start = False
                if waiting_for_session_start and current_epoch == current_resume_epoch_for_session:
                    # 找到该session的起始时间了
                    is_new_session_start = True
                    waiting_for_session_start = False
                
                # 新epoch，追加到列表（即使epoch相同也追加，保留所有训练记录）
                results.append({
                    'epoch': current_epoch,
                    'time': current_time,
                    'learning_rate': None,
                    'mAP': None,
                    'Rank-1': None,
                    'Rank-5': None,
                    'Rank-10': None,
                    'train_session': len([r for r in results if r.get('epoch') == current_epoch]) + 1,
                    'is_session_start': is_new_session_start  # 新增：标记是否为session起始
                })
                
                # 初始化当前epoch的学习率记录集合（如果不存在）
                if current_epoch not in epoch_lr_values:
                    epoch_lr_values[current_epoch] = set()
            
            # 提取学习率（从训练iteration中提取）
            lr_match = lr_pattern.search(line)
            if lr_match:
                lr_str = lr_match.group(1)
                try:
                    # 将科学计数法或小数转换为浮点数
                    if 'e' in lr_str.lower():
                        lr_val = float(lr_str)
                    else:
                        lr_val = float(lr_str)
                    
                    # 记录这个epoch中所有batch的学习率（用于检查是否保持不变）
                    # 我们需要从行中提取epoch信息：Epoch[1] Iteration[40/232]
                    epoch_match = re.search(r'Epoch\[(\d+)\]', line)
                    if epoch_match:
                        epoch_num = int(epoch_match.group(1))
                        if epoch_num not in epoch_lr_values:
                            epoch_lr_values[epoch_num] = set()
                        epoch_lr_values[epoch_num].add(lr_val)
                except ValueError:
                    pass
            
            # 提取指标
            if current_epoch is not None and results:
                # 找到当前epoch对应的结果（使用最新的一个，因为是追加的）
                target_result = results[-1]
                
                if target_result:
                    map_match = map_pattern.search(line)
                    if map_match:
                        target_result['mAP'] = float(map_match.group(1))
                    
                    rank1_match = rank1_pattern.search(line)
                    if rank1_match:
                        target_result['Rank-1'] = float(rank1_match.group(1))
                    
                    rank5_match = rank5_pattern.search(line)
                    if rank5_match:
                        target_result['Rank-5'] = float(rank5_match.group(1))
                    
                    rank10_match = rank10_pattern.search(line)
                    if rank10_match:
                        target_result['Rank-10'] = float(rank10_match.group(1))
                    
                    # 如果所有指标都提取完毕，重置current_epoch
                    # 注意：learning_rate可能为None，所以不检查它
                    required_fields = ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']
                    if all(target_result.get(field) is not None for field in required_fields):
                        # 如果该epoch有学习率记录
                        if current_epoch in epoch_lr_values and epoch_lr_values[current_epoch]:
                            lr_values = epoch_lr_values[current_epoch]
                            # 检查学习率在epoch内是否保持不变
                            if len(lr_values) == 1:
                                target_result['learning_rate'] = next(iter(lr_values))
                            else:
                                target_result['learning_rate'] = "Variable"
                        else:
                            target_result['learning_rate'] = None
                        current_epoch = None
                        
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        
    return results, breaks


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
    all_breaks = {}  # 存储每个模型的断点信息
    for log_info in rtx4090_logs:
        name = log_info['name']
        log_path = log_info['log_path']
        print(f"解析: {name}")
        
        results, breaks = parse_train_log(log_path)
        if results:
            all_results[name] = results
            all_breaks[name] = breaks
            print(f"  - 提取到 {len(results)} 个epoch的测试结果")
            if breaks:
                print(f"  - 检测到 {len(breaks)} 个断点")
        else:
            print(f"  - 未提取到有效数据")
    
    # 调试：检查学习率提取结果
    print(f"\n学习率提取调试信息:")
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r['mAP'] is not None]
        if valid_results:
            # 检查前几个epoch的学习率
            sample_results = valid_results[:5]
            for r in sample_results:
                lr = r.get('learning_rate')
                if lr is not None:
                    print(f"  {model_name} epoch {r['epoch']}: learning_rate = {lr}")
                else:
                    print(f"  {model_name} epoch {r['epoch']}: learning_rate = None")
            break  # 只检查第一个模型
    
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
        
        # 写入时间范围列表（包含断点信息）
        f.write("【训练先后顺序】\n\n")
        for i, (first_time, model_name, first, last) in enumerate(time_ranges, 1):
            f.write(f"{i}. {model_name}\n")
            
            results = all_results[model_name]
            valid_results = [r for r in results if r['time'] is not None]
            
            # 检查是否有断点
            breaks = all_breaks.get(model_name, [])
            
            if breaks:
                # 有断点，分段显示
                # 【修复】使用 prev_session_end_index 来确定段的结束epoch
                # break格式: (resume_epoch, last_time_before_break, prev_session_end_index, is_epoch1_restart)
                
                # 获取第一个验证结果的epoch和时间作为起始
                first_valid_epoch = valid_results[0]['epoch']
                
                # 段1: 从第一个epoch到第一个断点前一个session的结束
                first_break = breaks[0]
                resume_ep = first_break[0]  # 第一个恢复训练的epoch
                first_session_end_idx = first_break[2]  # 前一个session结束时的results索引
                # 【关键修复】从results中按索引找对应的记录
                first_session_end_record = results[first_session_end_idx] if first_session_end_idx >= 0 and first_session_end_idx < len(results) else None
                first_session_end_epoch = first_session_end_record['epoch'] if first_session_end_record else resume_ep - 1
                first_session_end_time = first_break[1]  # 前一个session结束的时间
                # 【优化】如果结束时间是None，使用该epoch记录的时间
                if first_session_end_time is None and first_session_end_record:
                    first_session_end_time = first_session_end_record.get('time')
                f.write(f"   段1: epoch {first_valid_epoch} ~ {first_session_end_epoch}\n")
                f.write(f"   时间: {first} ~ {first_session_end_time}\n")
                
                # 后续各段: 从每个断点的resume_epoch开始
                for j in range(len(breaks)):
                    last_break = breaks[j]
                    current_resume_ep = last_break[0]  # 当前段起始epoch
                    current_session_end_idx = last_break[2]  # 当前session结束时results索引
                    is_epoch1_restart = last_break[3]  # 是否是从 checkpoint 恢复后重新从 epoch 1 开始
                    
                    # 【关键修复】找到该 session 的真正起始时间
                    # 使用 is_session_start 标记
                    start_time = None
                    start_epoch_actual = None
                    for r in valid_results:
                        if r.get('is_session_start', False) and r['epoch'] == current_resume_ep:
                            start_time = r['time']
                            start_epoch_actual = r['epoch']
                            break
                    
                    # 如果没找到 is_session_start 标记，则按原逻辑找第一个匹配
                    if start_time is None:
                        for r in valid_results:
                            if r['epoch'] == current_resume_ep:
                                start_time = r['time']
                                start_epoch_actual = r['epoch']
                                break
                    
                    # 确定当前段的结束epoch
                    if j == len(breaks) - 1:
                        # 最后一个断点：段的结束是最后一个验证epoch
                        end_epoch = valid_results[-1]['epoch']
                        end_time_of段 = valid_results[-1]['time']
                    else:
                        # 中间断点：【关键修复】
                        # 当前段的结束 = 下一个断点发生前，第一个session的最后一条记录
                        # 即 breaks[j+1][2] - 1 对应的epoch（因为 breaks[j+1][2] 是检测到下一个断点时前一个session的结束位置）
                        next_break = breaks[j+1]
                        next_session_end_idx = next_break[2]  # 这是下一个断点检测时前一个session的结束位置
                        
                        if next_session_end_idx >= 0 and next_session_end_idx < len(results):
                            end_epoch_record = results[next_session_end_idx]
                            end_epoch = end_epoch_record['epoch']
                            end_time_of段 = end_epoch_record['time']
                        else:
                            # 回退到原来的逻辑
                            end_epoch = next_break[0] - 1
                            for r in valid_results:
                                if r['epoch'] == end_epoch:
                                    end_time_of段 = r['time']
                                    break
                            else:
                                end_time_of段 = last
                    
                    # 段号从2开始（因为段1已经写了）
                    segment_num = j + 2
                    f.write(f"   段{segment_num}: epoch {start_epoch_actual or current_resume_ep} ~ {end_epoch}\n")
                    f.write(f"   时间: {start_time} ~ {end_time_of段}\n")
            else:
                # 连续训练（无断点），也显示成段1格式
                first_epoch = valid_results[0]['epoch']
                last_epoch = valid_results[-1]['epoch']
                f.write(f"   段1: epoch {first_epoch} ~ {last_epoch}\n")
                f.write(f"   时间: {first} ~ {last}\n")
            f.write("\n")
        
        for model_name, results in all_results.items():
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"模型配置: {model_name}\n")
            f.write("=" * 70 + "\n")
            
            # 写入表头
            header = f"{'Epoch':<8} {'时间':<20} {'学习率':<15} {'mAP(%)':<10} {'Rank-1(%)':<12} {'Rank-5(%)':<12} {'Rank-10(%)':<12}\n"
            f.write(header)
            f.write("-" * 110 + "\n")
            
            # 写入每个epoch的结果，每25个epoch重复写表头
            for i, r in enumerate(results):
                if r['mAP'] is not None:
                    time_str = r['time'] if r['time'] else '-'
                    # 格式化学习率
                    if r.get('learning_rate') is None:
                        lr_str = '-'
                    elif r['learning_rate'] == "Variable":
                        lr_str = 'Variable'
                    else:
                        lr_val = r['learning_rate']
                        # 统一使用纯小数格式，保留6位小数，然后去除尾随的零和点
                        lr_str = f"{lr_val:.6f}".rstrip('0').rstrip('.')
                    
                    f.write(f"{r['epoch']:<8} {time_str:<20} {lr_str:<15} {r['mAP']:<10.1f} {r['Rank-1']:<12.1f} {r['Rank-5']:<12.1f} {r['Rank-10']:<12.1f}\n")
                    # 每25个epoch写一次表头（从第25个开始，即0索引24）
                    if (i + 1) % 25 == 0 and (i + 1) < len(results):
                        f.write("-" * 110 + "\n")
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
        
        # 在txt末尾写入差值和最小的前50个epoch统计表
        write_top50_pth_table(f, all_results, output_dir)
    
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
    
    # 在终端打印差值和最小的前50个epoch统计表
    print_top50_pth_table(all_results, output_dir)


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


def collect_all_pth_metrics(all_results, logs_dir):
    """收集所有epoch及其对应的指标数据
    
    返回格式:
    [
        {
            'model': 模型名称,
            'epoch': epoch编号,
            'pth_file': pth文件名(如果有),
            'mAP': mAP值,
            'Rank-1': Rank-1值,
            'diff_map': mAP与目标的差值,
            'diff_rank1': Rank-1与目标的差值,
            'diff_sum': 差值和,
            'diff_diff': |diff_map - diff_rank1|,
            'order': 读取顺序索引
        },
        ...
    ]
    """
    all_pth_data = []
    order_counter = 0  # 用于记录读取顺序
    
    # 获取所有模型的 pth epochs（用于标记是否有对应的 pth 文件）
    all_pth_epochs = {}
    for model_name in all_results.keys():
        all_pth_epochs[model_name] = set(find_pth_files(logs_dir, model_name))
    
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r['mAP'] is not None]
        if not valid_results:
            continue
        
        # 遍历所有 epoch，不依赖 pth 文件
        for r in valid_results:
            epoch = r['epoch']
            has_pth = epoch in all_pth_epochs.get(model_name, set())
            
            diff_map = r['mAP'] - TARGET_MAP
            diff_rank1 = r['Rank-1'] - TARGET_RANK1
            diff_sum = abs(diff_map) + abs(diff_rank1)
            diff_diff = abs(diff_map - diff_rank1)
            
            all_pth_data.append({
                'model': model_name,
                'epoch': epoch,
                'pth_file': f'transformer_checkpoint_{epoch}.pth' if has_pth else '-',
                'mAP': r['mAP'],
                'Rank-1': r['Rank-1'],
                'diff_map': diff_map,
                'diff_rank1': diff_rank1,
                'diff_sum': diff_sum,
                'diff_diff': diff_diff,
                'order': order_counter
            })
            order_counter += 1
    
    return all_pth_data


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


def strip_prefix(name):
    """去掉模型名称中的通用前缀"""
    prefixes = ['Ballshow_rtx4090_', 'BallShow_rtx4090_']
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def write_top50_pth_table(f, all_results, logs_dir):
    """在txt文件末尾写入差值和最小的前50个epoch统计表"""
    all_pth_data = collect_all_pth_metrics(all_results, logs_dir)
    
    if not all_pth_data:
        return
    
    # 排序规则:
    # 1. 首先按差值和(diff_sum)从小到大排序
    # 2. 如果差值和相等，按差值之间的差(diff_diff)从小到大排序
    # 3. 如果还相等，按mAP从大到小排序
    # 4. 如果还相等，按读取顺序(order)排序
    sorted_pth_data = sorted(all_pth_data, key=lambda x: (x['diff_sum'], x['diff_diff'], -x['mAP'], x['order']))
    
    # 取前50个
    top50 = sorted_pth_data[:50]
    
    f.write("\n\n")
    f.write("=" * 100 + "\n")
    f.write("【mAP和Rank-1离目标差值和最小的前50个epoch】\n")
    f.write("=" * 100 + "\n")
    f.write(f"目标: mAP >= {TARGET_MAP}%, Rank-1 >= {TARGET_RANK1}%\n")
    f.write("排序规则: 1.差值和最小 2.差值差最小 3.mAP最大 4.读取顺序\n\n")
    
    # 表头
    f.write(f"{'排名':^4}  {'模型名称':<40}  {'Epoch':^5}  {'mAP':^5}  {'Rank-1':^6}  {'mAP差值':^6}  {'R1差值':^6}  {'差值和':^6}  {'差值差':^6}  {'pth':^4}\n")
    f.write("-" * 108 + "\n")
    
    for rank, item in enumerate(top50, 1):
        # 格式化差值显示
        if item['diff_map'] >= 0:
            diff_map_str = f"+{item['diff_map']:.1f}%"
        else:
            diff_map_str = f"{item['diff_map']:.1f}%"
        
        if item['diff_rank1'] >= 0:
            diff_rank1_str = f"+{item['diff_rank1']:.1f}%"
        else:
            diff_rank1_str = f"{item['diff_rank1']:.1f}%"
        
        # pth 标记
        pth_mark = '有' if item['pth_file'] != '-' else '无'
        
        model = strip_prefix(item['model'])
        f.write(f"{rank:^4}  {model:<40}  {item['epoch']:^5}  {item['mAP']:^5.1f}  {item['Rank-1']:^6.1f}  {diff_map_str:^6}  {diff_rank1_str:^6}  {item['diff_sum']:>5.1f}%  {item['diff_diff']:>5.1f}%  {pth_mark:^4}\n")
    
    f.write("\n")
    f.write("=" * 100 + "\n")
    f.write("【说明】\n")
    f.write("- 差值 = 实际值 - 目标值\n")
    f.write("- 差值和 = |mAP差值| + |Rank-1差值|\n")
    f.write("- 差值差 = |mAP差值 - Rank-1差值|\n")
    f.write("- 差值和越小表示综合指标越接近目标\n")


def print_top50_pth_table(all_results, logs_dir):
    """在终端打印差值和最小的前50个epoch统计表"""
    all_pth_data = collect_all_pth_metrics(all_results, logs_dir)
    
    if not all_pth_data:
        return
    
    # 排序
    sorted_pth_data = sorted(all_pth_data, key=lambda x: (x['diff_sum'], x['diff_diff'], -x['mAP'], x['order']))
    top50 = sorted_pth_data[:50]
    
    print("\n")
    print("=" * 100)
    print("【mAP和Rank-1离目标差值和最小的前50个epoch】")
    print("=" * 100)
    print(f"目标: mAP >= {TARGET_MAP}%, Rank-1 >= {TARGET_RANK1}%")
    print("排序规则: 1.差值和最小 2.差值差最小 3.mAP最大 4.读取顺序\n")
    
    # 表头
    header = f"{'排名':^4}  {'模型名称':<40}  {'Epoch':^5}  {'mAP':^5}  {'Rank-1':^6}  {'mAP差值':^6}  {'R1差值':^6}  {'差值和':^6}  {'差值差':^6}  {'pth':^4}"
    print(header)
    print("-" * 108)
    
    for rank, item in enumerate(top50, 1):
        if item['diff_map'] >= 0:
            diff_map_str = f"+{item['diff_map']:.1f}%"
        else:
            diff_map_str = f"{item['diff_map']:.1f}%"
        
        if item['diff_rank1'] >= 0:
            diff_rank1_str = f"+{item['diff_rank1']:.1f}%"
        else:
            diff_rank1_str = f"{item['diff_rank1']:.1f}%"
        
        # pth 标记
        pth_mark = '有' if item['pth_file'] != '-' else '无'
        
        model = strip_prefix(item['model'])
        line = f"{rank:^4}  {model:<40}  {item['epoch']:^5}  {item['mAP']:^5.1f}  {item['Rank-1']:^6.1f}  {diff_map_str:^6}  {diff_rank1_str:^6}  {item['diff_sum']:>5.1f}%  {item['diff_diff']:>5.1f}%  {pth_mark:^4}"
        print(line)


def write_comparison_table(f, all_results, logs_dir):
    """在txt文件末尾写入pth文件与最优指标对照表格"""
    table_data = generate_pth_comparison_table(all_results, logs_dir)
    
    if not table_data:
        return
    
    f.write("\n\n")
    f.write("=" * 110 + "\n")
    f.write("【pth文件与最优指标对照表】\n")
    f.write("=" * 110 + "\n")
    f.write("\n自动分析各模型文件夹中的pth文件，与txt中记录的最优epoch进行对照\n")
    f.write("- [OK] 表示该epoch有对应的pth文件保存\n")
    f.write("- [NO] 表示该epoch没有对应的pth文件保存\n\n")
    
    # 表头 - 精确对齐
    f.write(f"{'模型名称':<35} {'pth文件':<35} {'最佳mAP':<15} {'最佳Rank-1':<15} {'最佳Rank-5':<15} {'最佳Rank-10':<15}\n")
    f.write("-" * 140 + "\n")
    
    for row in table_data:
        model = strip_prefix(row['model'])
        f.write(f"{model:<35} {row['pth_str']:<35} {row['best_map']:<15} {row['best_rank1']:<15} {row['best_rank5']:<15} {row['best_rank10']:<15}\n")
    
    # 统计信息
    f.write("\n" + "-" * 110 + "\n")
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
    print("=" * 110)
    print("【pth文件与最优指标对照表】")
    print("=" * 110)
    print("\n自动分析各模型文件夹中的pth文件，与txt中记录的最优epoch进行对照")
    print("- [OK] 表示该epoch有对应的pth文件保存")
    print("- [NO] 表示该epoch没有对应的pth文件保存\n")
    
    # 表头 - 精确对齐
    header = f"{'模型名称':<35} {'pth文件':<18} {'最佳mAP':<14} {'最佳Rank-1':<14} {'最佳Rank-5':<14} {'最佳Rank-10':<14}"
    print(header)
    print("-" * 110)
    
    for row in table_data:
        model = strip_prefix(row['model'])
        line = f"{model:<35} {row['pth_str']:<18} {row['best_map']:<14} {row['best_rank1']:<14} {row['best_rank5']:<14} {row['best_rank10']:<14}"
        print(line)
    
    # 统计信息
    print("\n" + "-" * 110)
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
