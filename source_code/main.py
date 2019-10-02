import os
import sys
import argparse
from lib.config import cfg, cfg_from_file, cfg_from_list
from trainer import Trainer
import shutil
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--config', dest='config', default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.config is not None:
        cfg_from_file(args.config)
        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR, exist_ok=True)
        shutil.copy2(args.config, cfg.ROOT_DIR)
    os.environ["TORCH_HOME"] = cfg.MODEL.PRETRAINED_PATH

    snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
    if os.path.exists(snapshot_folder):
        files = os.listdir(snapshot_folder)
        files = [k for k in files if k.endswith(".pth")]
        saved_epochs = [int(k.split("_")[-1].replace(".pth", "")) for k in files]
        last_epoch = np.max(saved_epochs)
        args.resume = last_epoch
        print("Modify resume to {}".format(args.resume))
    trainer = Trainer(args)
    if cfg.MODEL.SOURCE_ONLY == True:
        trainer.train_src_only()
    else:
        raise NotImplementedError
