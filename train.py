from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--resume", default="", help="path to checkpoint to resume from", type=str)
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
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
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

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            logger.info("Loaded checkpoint '{}'".format(args.resume))
            try:
                filename = os.path.basename(args.resume)
                if 'checkpoint' in filename:
                    epoch_str = filename.split('_')[-1].split('.')[0]
                else:
                    epoch_str = filename.split('_')[1].split('.')[0]
                start_epoch = int(epoch_str) + 1
                logger.info("Resuming from epoch {}".format(start_epoch))
            except:
                logger.info("Could not parse epoch from checkpoint filename, starting from epoch 1")
        else:
            logger.info("No checkpoint found at '{}'".format(args.resume))

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank,
        start_epoch
    )
