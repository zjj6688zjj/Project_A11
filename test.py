import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--test-all", action="store_true", help="Test all checkpoint files in output directory")
    parser.add_argument("--reranking", action="store_true", help="Use re-ranking for evaluation")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        # Fix for GBK encoding issues on Windows
        # Use yaml directly with explicit encoding
        import yaml
        
        # First load the config file
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f)
        except UnicodeDecodeError:
            # Try with GBK if UTF-8 fails
            try:
                with open(args.config_file, 'r', encoding='gbk') as f:
                    cfg_dict = yaml.safe_load(f)
            except UnicodeDecodeError:
                # Final fallback: binary read with latin-1
                with open(args.config_file, 'rb') as f:
                    content = f.read().decode('latin-1')
                    cfg_dict = yaml.safe_load(content)
        
        if cfg_dict:
            # Fix for YAML number type parsing
            def fix_yaml_types(value):
                """Recursively fix YAML parsed values to ensure correct types"""
                if isinstance(value, dict):
                    return {k: fix_yaml_types(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [fix_yaml_types(v) for v in value]
                elif isinstance(value, str):
                    # Try to convert string numbers to actual numbers
                    try:
                        # Handle scientific notation like "1e-4"
                        if 'e' in value.lower():
                            return float(value)
                        # Handle regular integers and floats
                        if '.' in value:
                            return float(value)
                        else:
                            return int(value)
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        return value
                else:
                    # Already correct type (int, float, bool, etc.)
                    return value
            
            # Apply type fixing
            cfg_dict = fix_yaml_types(cfg_dict)
            
            # Custom merge function that handles missing keys gracefully
            def merge_dict_into_cfg(cfg_dict, cfg_node, prefix=""):
                for key, value in cfg_dict.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        # Recursively merge nested dictionaries
                        if hasattr(cfg_node, key):
                            merge_dict_into_cfg(value, getattr(cfg_node, key), full_key)
                        else:
                            # If key doesn't exist, create it
                            from yacs.config import CfgNode
                            setattr(cfg_node, key, CfgNode(value))
                    else:
                        # Set scalar value
                        if hasattr(cfg_node, key):
                            setattr(cfg_node, key, value)
                        else:
                            # If key doesn't exist, create it (for new parameters like TEST.K1)
                            setattr(cfg_node, key, value)
            
            # Apply the custom merge
            merge_dict_into_cfg(cfg_dict, cfg)
    cfg.merge_from_list(args.opts)

    # 命令行参数优先，强制覆盖配置文件中的RE_RANKING设置
    if args.reranking:
        cfg.TEST.RE_RANKING = True

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # Fix encoding issue for reading config file
        try:
            with open(args.config_file, 'r', encoding='utf-8') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        except UnicodeDecodeError:
            try:
                with open(args.config_file, 'r', encoding='gbk') as cf:
                    config_str = "\n" + cf.read()
                    logger.info(config_str)
            except UnicodeDecodeError:
                # Skip config file display if encoding fails
                logger.info("Config file loaded (encoding details skipped due to charset issues)")
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 检查是否要测试所有pth文件
    test_all = args.test_all
    weight_path = cfg.TEST.WEIGHT

    if test_all:
        # 测试所有pth文件
        weight_files = []
        output_dir = cfg.OUTPUT_DIR

        # 查找所有transformer_checkpoint_*.pth文件
        for file in os.listdir(output_dir):
            if file.startswith('transformer_checkpoint_') and file.endswith('.pth'):
                weight_files.append(file)

        # 按epoch号排序
        weight_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        logger.info(f"Found {len(weight_files)} checkpoint files to test")
        for file in weight_files:
            logger.info(f"  - {file}")
    else:
        # 只测试指定的权重文件
        weight_files = [weight_path]

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    # 依次测试每个权重文件
    for weight_file in weight_files:
        if test_all:
            # 构建完整路径
            weight_path = os.path.join(cfg.OUTPUT_DIR, weight_file)

        logger.info(f"\nTesting with weight: {weight_path}")

        # 加载权重
        model.load_param(weight_path)

        if cfg.DATASETS.NAMES == 'VehicleID':
            for trial in range(10):
                train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
                rank_1, rank5 = do_inference(cfg,
                     model,
                     val_loader,
                     num_query,
                     use_reranking=args.reranking)
                if trial == 0:
                    all_rank_1 = rank_1
                    all_rank_5 = rank5
                else:
                    all_rank_1 = all_rank_1 + rank_1
                    all_rank_5 = all_rank_5 + rank5

                logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
            logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
        else:
           do_inference(cfg,
                     model,
                     val_loader,
                     num_query,
                     use_reranking=args.reranking)

