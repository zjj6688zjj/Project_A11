# -*- coding: utf-8 -*-
"""
自动读取 configs/BallShow/*.yml 文件，检查对应的 logs/ 文件夹是否存在
生成对应关系报告
"""

import os
import yaml
from datetime import datetime

def get_actual_logs_folder(yml_path):
    """从yml文件读取OUTPUT_DIR，返回实际的logs文件夹路径"""
    with open(yml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir = config.get('OUTPUT_DIR', '')
    
    # 如果 OUTPUT_DIR 已经是完整路径（logs/开头），直接使用
    if output_dir.startswith('logs/') or output_dir.startswith('logs\\'):
        return output_dir.replace('\\', '/')
    elif output_dir.startswith('./logs/'):
        return output_dir[2:].replace('\\', '/')
    elif output_dir.startswith('../logs/'):
        return output_dir[3:].replace('\\', '/')
    elif output_dir:
        # 如果只是目录名，加上 logs/ 前缀
        return 'logs/' + output_dir
    else:
        return None

def get_best_results_from_log(train_log_path):
    """从训练日志中提取最佳结果"""
    if not os.path.exists(train_log_path):
        return ""
    
    try:
        with open(train_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        best_map = 0
        best_rank1 = 0
        best_epoch = 0
        
        for line in lines:
            if "Test" in line and "mAP" in line:
                parts = line.strip().split()
                epoch = 0
                map_val = 0
                rank1_val = 0
                
                for i, part in enumerate(parts):
                    if "Epoch" in part and i + 1 < len(parts):
                        try:
                            epoch = int(parts[i + 1])
                        except:
                            pass
                    if "mAP" in part and i + 1 < len(parts):
                        try:
                            map_val = float(parts[i + 1])
                        except:
                            pass
                    if "Rank-1" in part and i + 1 < len(parts):
                        try:
                            rank1_val = float(parts[i + 1])
                        except:
                            pass
                
                if map_val > best_map:
                    best_map = map_val
                    best_rank1 = rank1_val
                    best_epoch = epoch
        
        if best_map > 0:
            return f"Epoch {best_epoch}, mAP {best_map:.1f}%, Rank-1 {best_rank1:.1f}%"
    except:
        pass
    
    return ""

def check_yml_pth_correspondence():
    configs_dir = "configs/BallShow"
    
    # 获取所有 yml 文件
    yml_files = [f for f in os.listdir(configs_dir) if f.endswith('.yml')]
    yml_files.sort()
    
    results = []
    
    for yml_file in yml_files:
        yml_name = os.path.splitext(yml_file)[0]  # 去掉 .yml 扩展名
        
        # 读取 yml 文件获取 OUTPUT_DIR
        yml_path = os.path.join(configs_dir, yml_file)
        output_dir = get_actual_logs_folder(yml_path)
        
        # 检查 logs 文件夹是否存在
        logs_path = output_dir if output_dir else yml_name
        folder_exists = os.path.isdir(logs_path)
        
        # 获取日志文件中的最佳结果
        best_results = ""
        if folder_exists:
            train_log_path = os.path.join(logs_path, "train_log.txt")
            best_results = get_best_results_from_log(train_log_path)
        
        # 获取 pth 文件数量
        pth_count = 0
        if folder_exists:
            pth_files = [f for f in os.listdir(logs_path) if f.endswith('.pth')]
            pth_count = len(pth_files)
        
        results.append({
            'yml_file': yml_file,
            'yml_name': yml_name,
            'output_dir': logs_path,
            'folder_exists': folder_exists,
            'pth_count': pth_count,
            'best_results': best_results
        })
    
    # 生成 TXT 文件 (configs/BallShow)
    generate_txt_file(results)
    
    # 终端也输出内容
    with open("configs/BallShow/yml_pth_correspondence.txt", 'r', encoding='utf-8') as f:
        print(f.read())

def generate_txt_file(results):
    """生成 TXT 格式的报告"""
    txt_file = "configs/BallShow/yml_pth_correspondence.txt"
    
    trained = [r for r in results if r['folder_exists']]
    not_trained = [r for r in results if not r['folder_exists']]
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("YML配置文件与Logs文件夹对应关系\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # 统计
        f.write("【统计】\n")
        f.write(f"总配置文件数: {len(results)}\n")
        f.write(f"已训练: {len(trained)}\n")
        f.write(f"未训练: {len(not_trained)}\n\n")
        
        # 已训练的配置文件
        f.write("=" * 70 + "\n")
        f.write("【已训练的配置文件】({}个)\n".format(len(trained)))
        f.write("-" * 70 + "\n")
        f.write("{:<6} {:<55} {:<45}\n".format('序号', 'YML文件', 'Logs文件夹路径'))
        f.write("-" * 120 + "\n")
        
        for i, r in enumerate(trained, 1):
            f.write("{:<6} {:<55} {:<45}\n".format(
                i, r['yml_file'], r['output_dir']))
        
        if not trained:
            f.write("{:<6} {:<55}\n".format('', '无'))
        
        f.write("\n")
        
        # 未训练的配置文件
        f.write("=" * 70 + "\n")
        f.write("【未训练的配置文件】({}个)\n".format(len(not_trained)))
        f.write("-" * 70 + "\n")
        f.write("{:<6} {:<55} {:<40}\n".format('序号', 'YML文件', 'Logs文件夹路径'))
        f.write("-" * 70 + "\n")
        
        for i, r in enumerate(not_trained, 1):
            f.write("{:<6} {:<55} {:<40}\n".format(
                i, r['yml_file'], r['output_dir']))

if __name__ == '__main__':
    check_yml_pth_correspondence()
