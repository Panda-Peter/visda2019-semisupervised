import pickle
import numpy as np
import os
from tqdm import tqdm
import argparse

load_prototypes_saved = True
save_prototypes = True

parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('--domain', type=str, default="clipart")
parser.add_argument("--model", type=str, default="")
parser.add_argument("--enable_prototype", type=int, default=1)
parser.add_argument("--prototye_acorss_model", type=int, default=0)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--source_only_proto", type=str, default="")

args = parser.parse_args()
enable_prototype = args.enable_prototype
domain = args.domain

total_files = []

data_dirs = [
    "experiments/0928_final_ensemble_results/0927_v3_full_set_results_multi_crop",
    "experiments/0928_final_ensemble_results/single_crop_old_data",
    "experiments/0928_final_ensemble_results/single_crop_baseline_ema",
    "experiments/0928_final_ensemble_results/single_crop_increase_resolution",
    "experiments/0928_final_ensemble_results/single_crop_increase_resolution_ema",
    "experiments/0928_final_ensemble_results/single_crop_increase_resolution_no_crop",
    "experiments/0928_final_ensemble_results/single_crop_increase_resolution_no_crop_ema"
]

if args.model != "":
    models_idx = args.model.strip().split(",")
    models_idx = [int(m) for m in models_idx]
    data_dirs = [data_dirs[k] for k in models_idx]

for data_dir in data_dirs:
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, m) for m in files]
    total_files.extend(files)

files = total_files

files = [m for m in files if m.endswith(".npy") and "source" not in m and "feature" not in m and "prototype" not in m]

files = sorted(files)

all_probs = {}

label_path = "../data/list/clipart_0924_pseudo_label_round3_pseudo_v3_train.txt"
labels = open(label_path).readlines()
total_names = [k.strip().split(" ")[0] for k in labels]
labels = [int(k.strip().split(" ")[1]) for k in labels]
labels = np.array(labels)


def src_feature_name_patter(sinlge_model):
    src_1 = sinlge_model.replace("target.npy", "source_feature.npy")
    src_2 = sinlge_model.replace(".npy", "_source_feature.npy")
    total_ = [src_1, src_2]
    total_ = list(set(total_))
    total_ = [m for m in total_ if "feature" in m]
    valid = [os.path.exists(m) for m in total_]
    valie_cnt = np.sum(valid)
    assert valie_cnt == 1, sinlge_model
    return [total_[j] for j in range(len(valid)) if valid[j]][0]


def target_feature_name_patter(sinlge_model):
    src_1 = sinlge_model.replace("target.npy", "target_feature.npy")
    src_2 = sinlge_model.replace(".npy", "_feature.npy")
    total_ = [src_1, src_2]
    total_ = list(set(total_))
    total_ = [m for m in total_ if "feature" in m]
    valid = [os.path.exists(m) for m in total_]
    valie_cnt = np.sum(valid)
    assert valie_cnt == 1, sinlge_model
    return [total_[j] for j in range(len(valid)) if valid[j]][0]


def load_prototypes(files_paths):
    files_paths = sorted(files_paths)
    files_paths = [m for m in files_paths if domain in m]
    total_paths = [src_feature_name_patter(sinlge_model) + "prototype2.npy" for sinlge_model in files_paths]
    prototypes = [np.load(m) for m in total_paths]
    inside_models = 7
    outsize_models = 7
    total_prototypes = []
    for j in range(0, inside_models):
        single_model = [prototypes[k] for k in [inside_models * m + j for m in range(0, outsize_models)]]
        single_model = np.stack(single_model, axis=0)
        total_prototypes.append(single_model)
    return total_prototypes


def add_prototype_score(sinlge_model, this_prototype=None):
    src_feature_name = src_feature_name_patter(sinlge_model)
    prototype_name = src_feature_name + "prototype2.npy"
    if load_prototypes_saved and os.path.exists(prototype_name):
        prototype = np.load(prototype_name)
    else:
        src_feature = np.load(src_feature_name)
        src_feature = src_feature / np.sqrt(np.sum(np.square(src_feature), axis=1, keepdims=True))
        prototype = []
        for k in tqdm(range(0, 345)):
            if args.source_only_proto:
                idx = [j for j in range(len(labels)) if labels[j] == k and "real" not in total_names[j]]
            else:
                idx = [j for j in range(len(labels)) if labels[j] == k]
            names = []
            leave_idx = []
            for j in idx:
                if total_names[j] not in names:
                    names.append(total_names[j])
                    leave_idx.append(j)
            sampled_feature = src_feature[idx]
            sampled_feature = np.mean(sampled_feature, axis=0)
            prototype.append(sampled_feature)
        prototype = np.array(prototype)
    if save_prototypes:
        np.save(prototype_name, prototype)

    if this_prototype is not None:
        prototype = this_prototype

    target_feature = np.load(target_feature_name_patter(sinlge_model))
    target_feature = target_feature / np.sqrt(np.sum(np.square(target_feature), axis=1, keepdims=True))
    if this_prototype is not None:
        scores_list = []
        for j in range(len(prototype.shape[0])):
            scores_ = np.dot(target_feature, np.transpose(prototype[j]))
            scores_list.append(scores_)
        scores = np.mean(np.stack(scores_list, axis=0), axis=0)
    else:
        scores = np.dot(target_feature, np.transpose(prototype))
    return scores, prototype


if args.prototye_acorss_model:
    print("Build prototype across models")
    all_prototype = load_prototypes(files)
files = [m for m in files if domain in m]
files = sorted(files)
for idx, m in tqdm(enumerate(files), total=len(files)):
    probs = np.load(m)
    if enable_prototype:
        if args.prototye_acorss_model:
            this_prototype = all_prototype[idx % len(all_prototype)]
            score, prototype = add_prototype_score(m, this_prototype)
        else:
            score, prototype = add_prototype_score(m)
        probs = (probs + score) / 2

    all_probs[m] = probs

efficent_probs = [m for m in all_probs.values()]

print("used model number is, ", len(efficent_probs))
efficent_probs_list = [np.array(m) for m in efficent_probs]
efficent_probs_list_array = np.stack(efficent_probs_list, axis=0)
efficent_probs_list_array = np.mean(efficent_probs_list_array, axis=0)
prediction = np.argmax(np.array(efficent_probs_list_array), axis=1).tolist()
if args.output != "":
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "{}_pred.txt".format(domain)), "w") as g:
        for k in prediction:
            g.write(str(k) + "\n")
