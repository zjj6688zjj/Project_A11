#!/usr/bin/env python3
"""
自动化测试脚本 - 测试三个4090配置文件的带和不带rerank性能
按照顺序: final → final153 → final212
每个文件夹先测不带rerank再测带rerank
"""

import subprocess
import time
import os

# 配置信息 - 按照要求的顺序: final → final153 → final212
configs = [
    {
        "name": "rtx4090_final",
        "config_file": "configs/BallShow/rtx4090_final.yml",
        "log_dir": "logs/BallShow_rtx4090_final",
        "model_count": 2  # transformer_checkpoint_153.pth, transformer_checkpoint_212.pth
    },
    {
        "name": "rtx4090_from_final153model", 
        "config_file": "configs/BallShow/rtx4090_from_final153model.yml",
        "log_dir": "logs/BallShow_rtx4090_from_final153model",
        "model_count": 1  # transformer_checkpoint_177.pth
    },
    {
        "name": "rtx4090_from_final212model",
        "config_file": "configs/BallShow/rtx4090_from_final212model.yml",
        "log_dir": "logs/BallShow_rtx4090_from_final212model",
        "model_count": 2  # transformer_checkpoint_230.pth, transformer_checkpoint_283.pth
    }
]

def run_test(command, test_name):
    """
    运行单个测试命令
    """
    print(f"\n{'='*60}")
    print(f"开始测试: {test_name}")
    print(f"命令: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        # 运行测试命令
        process = subprocess.run(command, capture_output=True, text=True)
        
        # 输出结果摘要
        if process.stdout:
            # 提取关键信息
            lines = process.stdout.split('\n')
            key_lines = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['mAP:', 'rank-1', 'validation results', 'testing with weight']):
                    key_lines.append(line.strip())
            
            if key_lines:
                print("关键输出:")
                for line in key_lines[-10:]:  # 显示最后10行关键信息
                    print(f"  {line}")
            else:
                print(f"输出摘要:\n{process.stdout[:300]}...")
                
        if process.stderr:
            print(f"错误:\n{process.stderr[:300]}...")
        
        print(f"返回码: {process.returncode}")
        
        if process.returncode == 0:
            print(f"✅ {test_name} 测试成功完成!")
        else:
            print(f"❌ {test_name} 测试失败!")
            
        return process.returncode
        
    except Exception as e:
        print(f"❌ {test_name} 测试出错: {e}")
        return -1

def main():
    """
    主函数：按照指定顺序依次运行测试
    """
    print("=" * 70)
    print("自动化测试脚本 - 三个4090配置文件带/不带rerank性能对比")
    print("=" * 70)
    print("测试顺序: rtx4090_final → rtx4090_from_final153model → rtx4090_from_final212model")
    print("每个文件夹测试顺序: 先不带rerank → 再带rerank")
    print("=" * 70)
    print("测试计划详情:")
    
    total_models = 0
    for i, config in enumerate(configs):
        print(f"{i+1}. {config['name']}: {config['model_count']}个模型 × 2种测试 = {config['model_count']*2}次测试")
        total_models += config['model_count'] * 2
    
    print(f"\n总计: {len(configs)}个配置文件，{total_models}次测试")
    print("=" * 70)
    
    results = []
    test_count = 0
    
    # 按照指定顺序测试每个配置文件
    for config_idx, config in enumerate(configs):
        print(f"\n{'#'*70}")
        print(f"测试文件夹 {config_idx+1}/{len(configs)}: {config['name']}")
        print(f"模型数量: {config['model_count']}个")
        print(f"配置文件: {config['config_file']}")
        print(f"日志目录: {config['log_dir']}")
        print(f"{'#'*70}")
        
        # 步骤1: 不带rerank的测试
        test_count += 1
        print(f"\n步骤 {test_count}: 不带rerank测试")
        
        no_rerank_cmd = [
            "python", "test.py", "--test-all",
            "--config_file", config['config_file']
        ]
        
        result1 = run_test(no_rerank_cmd, f"{config['name']}_without_rerank")
        results.append((f"{config['name']}_without_rerank", result1))
        
        # 等待5秒
        print("等待5秒...")
        time.sleep(5)
        
        # 步骤2: 带rerank的测试
        test_count += 1
        print(f"\n步骤 {test_count}: 带rerank测试")
        
        with_rerank_cmd = [
            "python", "test.py", "--reranking", "--test-all",
            "--config_file", config['config_file']
        ]
        
        result2 = run_test(with_rerank_cmd, f"{config['name']}_with_rerank")
        results.append((f"{config['name']}_with_rerank", result2))
        
        # 如果不是最后一个配置文件，等待10秒
        if config_idx < len(configs) - 1:
            print(f"\n等待10秒后开始下一个文件夹测试...")
            time.sleep(10)
    
    # 总结结果
    print(f"\n{'='*70}")
    print("测试总结报告:")
    print(f"{'='*70}")
    
    success_count = 0
    failed_tests = []
    
    for test_name, result_code in results:
        status = "✅ 成功" if result_code == 0 else "❌ 失败"
        print(f"{test_name}: {status}")
        if result_code == 0:
            success_count += 1
        else:
            failed_tests.append(test_name)
    
    print(f"\n总计: {success_count}/{len(results)} 次测试成功")
    
    if failed_tests:
        print(f"失败的测试: {', '.join(failed_tests)}")
    
    # 性能对比建议
    print(f"\n{'='*70}")
    print("性能对比建议:")
    print(f"{'='*70}")
    print("测试结果存储在以下日志文件中:")
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  日志文件: {config['log_dir']}/test_log.txt")
        print(f"  对比方法: 查看同一模型带/不带rerank的性能差异")
        print(f"  预期结果: rerank通常会使mAP下降4-8%")
    
    print(f"\n{'='*70}")
    if success_count == len(results):
        print("🎉 所有测试都成功完成!")
        print("✅ 请检查各日志文件查看详细性能对比")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()