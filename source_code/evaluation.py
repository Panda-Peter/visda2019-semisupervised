import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from lib.config import cfg


def eval(epoch, name, test_loader, netG, netE, return_probs=False, return_features=False):
    if epoch == 0:
        return None, None

    netG.eval()
    netE.eval()

    correct = 0
    tick = 0
    subclasses_correct = np.zeros(cfg.MODEL.CLASS_NUM)
    subclasses_tick = np.zeros(cfg.MODEL.CLASS_NUM)
    n_samples = test_loader.dataset.__len__()
    probs = torch.zeros(n_samples, cfg.MODEL.CLASS_NUM).cuda()
    features = torch.zeros(n_samples, cfg.MODEL.IN_DIM).cuda()

    results = []
    with torch.no_grad():
        index = 0
        for (imgs, labels) in tqdm.tqdm(test_loader):
            imgs = imgs.cuda()
            batch_size = imgs.size(0)

            imgs_shape = imgs.shape

            if len(imgs_shape) == 5:
                bs, cn, c, h, w = imgs.shape
                imgs = imgs.view(bs * cn, c, h, w)
            _, _unsup_pool5_out = netG(imgs)
            _, _unsup_logits_out = netE(_unsup_pool5_out)
            if len(imgs_shape) == 5:
                _unsup_logits_out = _unsup_logits_out.view(bs, cn, -1).mean(dim=1)
                _unsup_pool5_out = _unsup_pool5_out.view(bs, cn, -1).mean(dim=1)

            pred = F.softmax(_unsup_logits_out, dim=1)
            probs[index:index + batch_size] = pred.data
            features[index:index + batch_size] = _unsup_pool5_out.data

            index += batch_size
            pred = pred.cpu().numpy().argmax(axis=1)
            results += list(pred)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    correct += 1
                    subclasses_correct[pred[i]] += 1
                tick += 1
    correct = correct * 1.0 / tick
    subclasses_result = np.divide(subclasses_correct, subclasses_tick)
    mean_class_acc = subclasses_result.mean()
    zeros_num = subclasses_result[subclasses_result == 0].shape[0]

    mean_acc_str = "*** Epoch {:d}, {:s}; mean class acc = {:.2%}, overall = {:.2%}, missing = {:d}".format( \
        epoch, name, mean_class_acc, correct, zeros_num)
    if return_features:
        return mean_acc_str, results, probs.detach().cpu().numpy(), features.detach().cpu().numpy()
    if return_probs:
        return mean_acc_str, results, probs.detach().cpu().numpy()
    else:
        return mean_acc_str, results
