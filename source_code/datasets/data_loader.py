import os
import datasets.list_loader as list_loader
import datasets.online_loader as online_loader
import torch
from torchvision import transforms
import lib.utils as utils
from lib.config import cfg
import torch.distributed as dist
import samplers.distributed
from datasets.augmentation import get_transform


def make_src_dataset(src_list_path):
    root = cfg.DATA_LOADER.DATA_ROOT
    transform = transforms.Compose(get_transform(cfg.AUG.ADVANCE))

    if cfg.DATA_LOADER.SOURCE_TYPE == 'online':
        paths, labels = utils.loadlines(os.path.join(root, 'list', src_list_path))
        image_set = online_loader.OnlineLoader(root, paths, labels, transform)
        return image_set
    else:
        image_set = list_loader.ListLoader(root, os.path.join(root, 'list', src_list_path), transform)
        return image_set


def make_target_dataset(paths=None, labels=None, teacher_mode=False, online=False):
    if online:
        root = cfg.DATA_LOADER.DATA_ROOT
        transform = transforms.Compose(get_transform(cfg.AUG.ADVANCE))
        image_set = online_loader.OnlineLoader(root, paths, labels, transform, teacher_mode=teacher_mode)
        return image_set
    else:
        root = cfg.DATA_LOADER.DATA_ROOT
        transform = transforms.Compose(get_transform(cfg.AUG.ADVANCE))

        image_set = list_loader.ListLoader(root, paths,
                                           transform)
        return image_set


def make_online_dataloader(img_label_list, batch_size):
    root = cfg.DATA_LOADER.DATA_ROOT
    transform = transforms.Compose(get_transform(cfg.AUG.ADVANCE))
    image_set = list_loader.ListLoader(root, img_label_list,
                                       transform)
    loader = torch.utils.data.DataLoader(image_set, batch_size=batch_size,
                                         shuffle=cfg.DATA_LOADER.SHUFFLE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                         drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)

    return loader


def make_src_dataloader(distributed, image_set):
    if cfg.DATA_LOADER.SOURCE_TYPE == 'online':
        gpu_num = dist.get_world_size() if distributed else 1
        assert cfg.TRAIN.BATCH_SIZE % cfg.DATA_LOADER.CLASS_NUM_PERBATCH == 0
        imgs_per_cls = cfg.TRAIN.BATCH_SIZE // cfg.DATA_LOADER.CLASS_NUM_PERBATCH
        cls_info = image_set.sample_cls(cfg.DATA_LOADER.ITER, cfg.DATA_LOADER.CLASS_NUM_PERBATCH * gpu_num,
                                        cfg.MODEL.CLASS_NUM)
        index = image_set.samples(imgs_per_cls, cls_info)
        index = image_set.shuffle_index(index, cfg.TRAIN.BATCH_SIZE, gpu_num)

        if distributed:
            index = torch.tensor(index, device="cuda")
            torch.distributed.broadcast(index, 0)
            index = index.data.cpu().numpy().tolist()

        sampler = samplers.distributed.DistributedSamplerOnline(image_set, batch_size=cfg.TRAIN.BATCH_SIZE, index=index,
                                                                distributed=distributed)

        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                             drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                             sampler=sampler)
    else:
        cls_info = None
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=cfg.DATA_LOADER.SHUFFLE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                             drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
    return cls_info, loader


def make_target_dataloader(distributed, paths, labels, cls_info, teacher_mode=False):
    if (cfg.DATA_LOADER.TARGET_TYPE == 'online') and (labels is not None) and (cls_info is not None):
        image_set = make_target_dataset(paths, labels, teacher_mode, online=True)

        assert cfg.TRAIN.BATCH_SIZE % cfg.DATA_LOADER.CLASS_NUM_PERBATCH == 0
        imgs_per_cls = cfg.TRAIN.BATCH_SIZE // cfg.DATA_LOADER.CLASS_NUM_PERBATCH
        index = image_set.samples(imgs_per_cls, cls_info)
        gpu_num = dist.get_world_size() if distributed else 1
        index = image_set.shuffle_index(index, cfg.TRAIN.BATCH_SIZE, gpu_num)

        if distributed:
            index = torch.tensor(index, device="cuda")
            torch.distributed.broadcast(index, 0)
            index = index.data.cpu().numpy().tolist()

        sampler = samplers.distributed.DistributedSamplerOnline(image_set, batch_size=cfg.TRAIN.BATCH_SIZE, index=index,
                                                                distributed=distributed)
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                             drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                             sampler=sampler)
        return loader
    else:
        image_set = make_target_dataset(paths=paths, online=False)
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TEST.BATCH_SIZE,
                                             shuffle=cfg.DATA_LOADER.SHUFFLE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                             drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
        return loader


def load_test(target_root, test_label):
    if cfg.AUG.MULTI_CROP_TEST:
        transform = transforms.Compose([
            transforms.Resize(cfg.AUG.RESIZE),
            transforms.TenCrop(cfg.AUG.TEST_CROP),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(
                lambda crops: torch.stack([transforms.Normalize(cfg.MEAN, cfg.STD)(crop) for crop in crops]))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(cfg.AUG.RESIZE),
            transforms.CenterCrop(cfg.AUG.TEST_CROP),
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN, cfg.STD)
        ])
    if not os.path.exists(test_label):
        return None

    image_set = list_loader.ListLoader(target_root, test_label, transform)
    loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TEST.BATCH_SIZE,
                                         shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    return loader
