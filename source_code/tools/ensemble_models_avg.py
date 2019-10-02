import pickle
import numpy as np
import os
from tqdm import tqdm
import glob
import sys

data_dir = sys.argv[1]
domain = sys.argv[2]
outuput = sys.argv[3]
files = glob.glob(data_dir + "/**/*.npy")
files = sorted(files)
files = [m for m in files if domain in m ]
all_probs = {}
print("Fusion {} models".format(len(files)))
for m in tqdm(files):
    probs = np.load(m)
    all_probs[m] = probs

efficent_probs = [v for k, v in all_probs.items()]
print(len(efficent_probs))
efficent_probs = [np.array(m) for m in efficent_probs]
efficent_probs = np.stack(efficent_probs, axis=0)
efficent_probs = np.mean(efficent_probs, axis=0)
prediction = np.argmax(np.array(efficent_probs), axis=1).tolist()
if not os.path.exists(outuput):
    os.makedirs(outuput, exist_ok=True)
with open(os.path.join(outuput, "{}_pred.txt".format(domain)), "w")as g:
    for k in prediction:
        g.write(str(k) + "\n")
