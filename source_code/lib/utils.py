import os
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from lib.config import cfg


def loadlines(path):
    paths = []
    labels = []
    with open(path, 'r') as fid:
        for line in fid:
            data = line.strip().split(' ')
            paths.append(data[0])
            labels.append(int(data[1]))
    return paths, labels


def sync_labels(label):
    g_labels = [torch.ones_like(label) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(g_labels, label)
    gf_labels = torch.cat(g_labels, dim=0)
    return gf_labels


def sync_tensor(vec):
    g_vec = [torch.ones_like(vec) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(g_vec, vec)
    gf_vec = vec

    for i in range(0, dist.get_rank()):
        gf_vec = torch.cat((g_vec[dist.get_rank() - i - 1].detach(), gf_vec), dim=0)

    for i in range(dist.get_rank() + 1, dist.get_world_size()):
        gf_vec = torch.cat((gf_vec, g_vec[i].detach()), dim=0)
    return gf_vec


def broadcast_tensor(vec):
    torch.distributed.broadcast(vec, 0)
    vec = vec.data.cpu().numpy()
    return vec


def broadcast_numpy(vec):
    vec = torch.tensor(vec, device="cuda")
    torch.distributed.broadcast(vec, 0)
    vec = vec.data.cpu().numpy()
    return vec


def load_trg_plabels():
    if len(cfg.DATA_LOADER.TRG_PSEUDOLABELS) == 0:
        return None, None
    else:
        pdata = pickle.load(open(cfg.DATA_LOADER.TRG_PSEUDOLABELS, 'rb'), encoding='bytes')
        probs = pdata['probs']
        plabels = pdata['labels']
        return probs, plabels

def two_type_ac(pred, labels):
    if isinstance(pred, torch.Tensor): pred = pred.cpu().numpy()
    if isinstance(pred, torch.Tensor): labels = labels.cpu().numpy()
    ac = pred == labels
    per_cls_ac = []
    labels_set = set(labels.tolist())
    labels_set_num = len(labels_set)
    for k in labels_set:
        per_cls = np.sum(ac * (labels == k)) / (np.sum(labels == k) + 1e-10)
        per_cls_ac.append(per_cls)
    return np.mean(ac), np.mean(per_cls_ac)
