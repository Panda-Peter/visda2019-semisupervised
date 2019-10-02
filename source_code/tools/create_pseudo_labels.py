import os
import numpy as np
import sys

round_version = sys.argv[1]
domain = sys.argv[2]

if round_version == "1":
    base_folder = ["experiments/all_predictions/0920_noon/multi_crop_results"]
    output_name = "{}_0920_pseudo_label_round1_pseudo_v2_train.txt".format(domain)
elif round_version == "2":
    base_folder = ["experiments/all_predictions/0922_night/multi_crop_results"]
    output_name = "{}_0923_pseudo_label_round2_pseudo_v2_train.txt".format(domain)
elif round_version == "3":
    base_folder = [
        "experiments/all_predictions/0922_night/multi_crop_results",
        "experiments/all_predictions/0924_noon/multi_crop_results"
    ]
    output_name = "{}_0924_pseudo_label_round3_pseudo_v3_train.txt".format(domain)
else:
    raise NotImplementedError

model_path = []

for m in base_folder:
    files = os.listdir(m)
    model_path.extend([os.path.join(m, k) for k in files])

model_path = [m for m in model_path if domain in m and "npy" in m]

prediction = [np.load(m) for m in model_path]
prediction = np.mean(np.stack(prediction, axis=0), axis=0)
score = np.max(prediction, axis=1)
label = np.argmax(prediction, axis=1)
conf_thres = 0.7
keepd_idx = score > conf_thres

top_k_indices = []

idx = [k for k in range(len(keepd_idx)) if keepd_idx[k]]
idx_set = set(idx)

label_to_ids = {}
for i, k in enumerate(label.tolist()):
    if k not in label_to_ids:
        label_to_ids[k] = []
    if i in idx_set:
        label_to_ids[k].append([i, score[i], k])

for cls_id in range(0, prediction.shape[1]):
    scores = prediction[:, cls_id]
    top_k = np.argsort(-1. * scores)
    top_k = top_k[0:10]
    prediction[top_k, :] = 0.0
    if cls_id not in label_to_ids:
        label_to_ids[cls_id] = []
    for k in top_k:
        label_to_ids[cls_id].append([k, prediction[k, cls_id], cls_id])

max_num = 10000
min_num = 20
total_idx = []
total_label = []
for k, v in label_to_ids.items():
    sampled_label = []
    if len(v) > max_num:
        sampled = sorted(v, key=lambda x: -1. * x[1])
        sampled_label = [j[2] for j in sampled[:max_num]]
        sampled = [j[0] for j in sampled[:max_num]]
    else:
        if len(v) == 0:
            continue
        while len(v) < min_num:
            v = v + v
        sampled = [j[0] for j in v]
        sampled_label = [j[2] for j in v]
    assert len(sampled) >= min_num and len(sampled) <= max_num
    total_idx.extend(sampled)
    total_label.extend(sampled_label)

label = np.array(total_label)
src_file = open("../data/list/{}_unl.txt".format(domain)).readlines()
src_file = [m.strip().split(" ")[0] for m in src_file]
src_file = [src_file[k] for k in total_idx]
src1 = open("../data/list/real_{}_balance_train.txt".format(domain)).readlines()
if not os.path.exists(os.path.dirname("../data/list/{}".format(output_name))):
    os.makedirs(os.path.dirname("../data/list/{}".format(output_name)))

with open("../data/list/{}".format(output_name), "w") as g:
    for k, v in zip(src_file, label):
        src1.append(k + " " + str(v) + "\n")
    np.random.shuffle(src1)
    for k in src1:
        g.write(k.strip() + "\n")
