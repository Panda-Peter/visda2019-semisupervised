import os
import sys
import argparse
from lib.config import cfg, cfg_from_file, cfg_from_list
from trainer import Trainer
import shutil
import numpy as np
import evaluation
import torch.distributed as dist
import torch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--config', dest='config', default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--netG_model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--multi_crop", type=int, default=0)
    parser.add_argument("--save_feature", type=int, default=0)
    parser.add_argument("--eval_domain", type=str, default="target")
    parser.add_argument("--test_crop_size", type=int, default=-1)
    parser.add_argument("--test_resize_size", type=int, default=-1)
    parser.add_argument("--test_bs", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    args.resume = -1
    if args.config is not None:
        cfg_from_file(args.config)
    os.environ["TORCH_HOME"] = cfg.MODEL.PRETRAINED_PATH
    if args.multi_crop:
        cfg.AUG.MULTI_CROP_TEST = True
    if args.test_crop_size > 0:
        cfg.AUG.TEST_CROP = [args.test_crop_size, args.test_crop_size]
    if args.test_resize_size > 0:
        cfg.AUG.RESIZE = [args.test_resize_size, args.test_resize_size]
    if args.test_bs > 0:
        cfg.TEST.BATCH_SIZE = args.test_bs
    trainer = Trainer(args)
    netG_dict = torch.load(args.netG_model_path,
                           map_location=lambda storage, loc: storage)
    current_state = trainer.netG.state_dict()
    keys = list(current_state.keys())
    for key in keys:
        current_state[key] = netG_dict[key]
    trainer.netG.load_state_dict(current_state)
    netE_dict = torch.load(args.netG_model_path.replace("netG", "netE"),
                           map_location=lambda storage, loc: storage)
    current_state = trainer.netE.state_dict()
    keys = list(current_state.keys())
    for key in keys:
        current_state[key] = netE_dict[key]
    trainer.netE.load_state_dict(current_state)
    if args.eval_domain == "target":
        eval_domain = trainer.target_val_loader_eval_mode
        name = cfg.DATA_LOADER.TARGET + "_VAL"
    elif args.eval_domain == "source":
        eval_domain = trainer.src_train_loader_eval_mode
        name = cfg.DATA_LOADER.SOURCE + "_TRAIN"
    else:
        print("Not valid domain: {}".format(args.eval_domain))
        exit()
    if args.save_feature:
        mean_acc, preds, probs, features = evaluation.eval(1, name,
                                                           eval_domain,
                                                           trainer.netG, trainer.netE, return_probs=True,
                                                           return_features=True)
    else:
        mean_acc, preds, probs = evaluation.eval(1, name,
                                                 eval_domain,
                                                 trainer.netG, trainer.netE, return_probs=True,
                                                 return_features=False)
    if mean_acc is not None:
        if (trainer.distributed == False) or (dist.get_rank() == 0):
            trainer.logger.info(mean_acc)
            if not os.path.exists(os.path.dirname(args.output_path)):
                os.makedirs(os.path.dirname(args.output_path))
            with open(args.output_path, 'w') as fid:
                for v in preds:
                    fid.write(str(v) + '\n')
            np.save(args.output_path.replace(".txt", ".npy"), probs)
            if args.save_feature:
                np.save(args.output_path.replace(".txt", "_feature.npy"), features)
