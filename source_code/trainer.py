import os
import sys
import pprint
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import lib.utils as utils
from lib.config import cfg
import datasets.data_loader as data_loader
import time

import models
import losses
import evaluation

import torch.distributed as dist
import datetime


class AccuracyTracker():
    def __init__(self):
        self.training_acs = []
        self.moving_range = 100

    def trace_training_accuracy(self, logits, labels):
        logits_argmax = torch.argmax(logits, dim=1)
        ac = torch.eq(logits_argmax, labels).type(torch.float).mean().item()
        self.training_acs.append(ac)
        self.training_acs = self.training_acs[-self.moving_range:]

    def get_current_ac(self):
        return np.mean(self.training_acs)


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.setup_logging()
        self.setup_loader()
        self.init_network()
        self.iteration = 0
        self.total_iteration = -1
        self.start_time = time.time()
        self.last_display_time = self.start_time

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_loader(self):
        # data split
        # source: train
        # target: unlabelled, labeled, validation
        # make test loader for both source and target domain
        self.src_train_loader_eval_mode = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
                                                                os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list',
                                                                             cfg.DATA_LOADER.SOURCE + "_train.txt"))

        self.src_val_loader_eval_mode = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
                                                              os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list',
                                                                           cfg.DATA_LOADER.SOURCE + "_val.txt"))

        self.target_val_loader_eval_mode = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
                                                                 os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list',
                                                                              cfg.DATA_LOADER.TARGET + "_val.txt"))

        self.target_unlabeled_loader_eval_mode = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
                                                                       os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list',
                                                                                    cfg.DATA_LOADER.TARGET + "_unl.txt"))

        self.target_labeled_loader_eval_mode = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
                                                                     os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list',
                                                                                  cfg.DATA_LOADER.TARGET + "_labeled.txt"))

        # make src data loader
        self.src_image_set = data_loader.make_src_dataset(cfg.DATA_LOADER.SOURCE + "_train.txt")
        cls_info, self.src_train_loader = data_loader.make_src_dataloader(self.distributed, self.src_image_set)

        self.target_unlabeled_loader = data_loader.make_target_dataloader(self.distributed,
                                                                          paths=os.path.join(cfg.DATA_LOADER.DATA_ROOT,
                                                                                             'list',
                                                                                             cfg.DATA_LOADER.TARGET + '_unl.txt'),
                                                                          labels=None, cls_info=None,
                                                                          teacher_mode=False)

        self.target_labeled_loader = data_loader.make_target_dataloader(self.distributed,
                                                                        paths=os.path.join(cfg.DATA_LOADER.DATA_ROOT,
                                                                                           'list',
                                                                                           cfg.DATA_LOADER.TARGET + '_labeled.txt'),
                                                                        labels=None, cls_info=None,
                                                                        teacher_mode=False)

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)

        if epoch % cfg.SOLVER.SNAPSHOT_ITERS == 0:
            torch.save(self.netG.state_dict(), self.snapshot_path("netG", epoch))
            torch.save(self.netE.state_dict(), self.snapshot_path("netE", epoch))

    def load_checkpoint(self, netG, netE):
        if self.args.resume > 0:
            netG_dict = torch.load(self.snapshot_path("netG", self.args.resume),
                                   map_location=lambda storage, loc: storage)
            current_state = netG.state_dict()
            keys = list(current_state.keys())
            for key in keys:
                current_state[key] = netG_dict['module.' + key]
            netG.load_state_dict(current_state)

            netE_dict = torch.load(self.snapshot_path("netE", self.args.resume),
                                   map_location=lambda storage, loc: storage)
            current_state = netE.state_dict()
            keys = list(current_state.keys())
            for key in keys:
                current_state[key] = netE_dict['module.' + key]
            netE.load_state_dict(current_state)

    def init_network(self):
        netG = models.__dict__[cfg.MODEL.NET](pretrained=True)
        netE = models.classifier.Classifier(class_num=cfg.MODEL.CLASS_NUM, distributed=self.distributed).cuda()
        self.load_checkpoint(netG, netE)
        if self.distributed:
            sync_netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG)
            sync_netE = nn.SyncBatchNorm.convert_sync_batchnorm(netE)
            self.netG = torch.nn.parallel.DistributedDataParallel(sync_netG.to(self.device),
                                                                  device_ids=[self.args.local_rank],
                                                                  output_device=self.args.local_rank)
            self.netE = torch.nn.parallel.DistributedDataParallel(sync_netE.to(self.device),
                                                                  device_ids=[self.args.local_rank],
                                                                  output_device=self.args.local_rank)
        else:
            self.netG = torch.nn.DataParallel(netG).cuda()
            self.netE = torch.nn.DataParallel(netE).cuda()

        self.optim = models.optimizer.Optimizer(self.netG, self.netE)

        if cfg.LOSSES.LABEL_SMOOTH > 0:
            self.cross_ent = losses.create('SmoothCrossEntropy').cuda()
        else:
            self.cross_ent = losses.create('CrossEntropy').cuda()

    def eval(self, epoch):
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            if (self.distributed == False) or (dist.get_rank() == 0):
                os.mkdir(result_folder)

        loaders = []
        names = []
        if self.src_val_loader_eval_mode is not None:
            loaders.append(self.src_val_loader_eval_mode)
            names.append(cfg.DATA_LOADER.SOURCE + "_VAL")
        if self.target_val_loader_eval_mode is not None:
            loaders.append(self.target_val_loader_eval_mode)
            names.append(cfg.DATA_LOADER.TARGET + "_VAL")

        for i, loader in enumerate(loaders):
            mean_acc, preds = evaluation.eval(epoch, names[i], loader, self.netG, self.netE)
            if mean_acc is not None:
                if (self.distributed == False) or (dist.get_rank() == 0):
                    self.logger.info(mean_acc)
                    with open(os.path.join(result_folder, str(epoch) + '_' + names[i] + '.txt'), 'w') as fid:
                        for v in preds:
                            fid.write(str(v) + '\n')

    def train_mode(self):
        self.netG.train(mode=True)
        self.netE.train(mode=True)

    def display(self, epoch, iteration, losses, loss_arr, loss_w):
        # current_iteration = epoch * len(self.src_train_loader) + iteration
        total_time_used = time.time() - self.start_time
        total_time_used = str(datetime.timedelta(seconds=int(total_time_used)))
        if (self.distributed == True) and (dist.get_rank() != 0):
            return

        if iteration % cfg.SOLVER.DISPLAY == 0:
            time_since_last_display = time.time() - self.last_display_time
            self.last_display_time = time.time()
            current_speed = cfg.SOLVER.DISPLAY / time_since_last_display
            estimated_remaing_time = (self.total_iteration - iteration) / current_speed
            estimated_remaing_time = str(datetime.timedelta(seconds=int(estimated_remaing_time)))

            self.logger.info(
                "Epoch " + str(epoch) + ', Iteration ' + str(iteration % len(self.src_train_loader)) + "/" + str(
                    len(self.src_train_loader)) + ', lr = ' + str(
                    self.optim.get_lr()) + ', loss = ' + "%0.5f" % (
                    losses.data.cpu().numpy()) + ', cost-time = ' + str(
                    total_time_used) + ', speed = ' + "%0.2f" % (
                    current_speed) + 'fps, remaining = ' + str(estimated_remaing_time))
            for lidx, losses in enumerate(loss_arr):
                loss_name = losses[0]
                loss_value = losses[1]
                self.logger.info('  ' + loss_name + ' = ' + str(loss_value) \
                                 + ' (* ' + str(loss_w[lidx]) + ' = ' + str(loss_value * loss_w[lidx]) + ')')

    def train_pseudo_label(self):
        raise NotImplementedError

    def train_src_only(self):
        start_epoch = self.args.resume if self.args.resume > 0 else 0
        start_epoch = cfg.SOLVER.START_EPOCH if cfg.SOLVER.START_EPOCH >= 0 else start_epoch
        if start_epoch > 0:
            self.logger.info("Resume training from epoch: {}".format(start_epoch))
            self.logger.info("Update learning rate due to resuming")
            self.optim.step(start_epoch)

        self.total_iteration = (cfg.SOLVER.MAX_EPOCH - start_epoch) * len(self.src_train_loader)
        self.ac_tracker = AccuracyTracker()
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            self.eval(epoch)
            self.train_mode()

            for imgs, labels in self.src_train_loader:
                imgs = imgs.cuda()
                labels = labels.cuda()
                self.optim.zero_grad()
                _, sup_pool5_out = self.netG(imgs)
                _, sup_logits_out = self.netE(sup_pool5_out)
                if self.distributed:
                    labels = utils.sync_labels(labels)
                    sup_logits_out = utils.sync_tensor(sup_logits_out)
                loss_arr = []
                loss_w = []
                # source cross entropy loss
                loss, loss_info = self.cross_ent(sup_logits_out, labels)
                self.ac_tracker.trace_training_accuracy(sup_logits_out, labels)
                loss_arr.append(loss_info)
                loss_w.append(cfg.LOSSES.CROSS_ENT_WEIGHT)
                loss_arr.append(("02. accuracy", self.ac_tracker.get_current_ac()))
                loss_w.append(1.)
                self.display(epoch, self.iteration, loss, loss_arr, loss_w)

                if self.distributed:
                    loss *= dist.get_world_size()

                loss.backward()
                self.optim.step(epoch)
                self.iteration += 1

            self.save_model(epoch + 1)
            _, self.src_train_loader = data_loader.make_src_dataloader(self.distributed, self.src_image_set)
