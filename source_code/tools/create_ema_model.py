import os

files = os.listdir("./")
files = [m for m in files if "netG" in m]
files_to_pairs = {}
for m in files:
    net, epoch = m.strip().split("net")
    if net not in files_to_pairs:
        files_to_pairs[net] = []
    files_to_pairs[net].append(m)
for key, value in files_to_pairs.items():
    assert len(value) == 3
    print(key, value)
import torch

for key, value in files_to_pairs.items():
    tensors = [torch.load(m) for m in value]
    tensor_lst = {}
    for m in tensors[0].keys():
        sub_tensors = [j[m] for j in tensors]
        sub_tensors = sum(sub_tensors) / len(sub_tensors)
        tensor_lst[m] = sub_tensors
    print("Finish key {}".format(key))
    torch.save(tensor_lst, key + "ema.pth")
