import numpy as np
import sys
target_domain = sys.argv[1]

# target_domain = "painting"
# target_domain = "clipart"

src1 = open("../data/list/real_train.txt").readlines()
src2 = open("../data/list/{}_labeled.txt".format(target_domain)).readlines()

src2 = src2 * 10

src = src1 + src2

np.random.shuffle(src)
with open("../data/list/real_{}_balance_train.txt".format(target_domain), "w") as g:
    for k in src:
        g.write(k.strip() + "\n")
