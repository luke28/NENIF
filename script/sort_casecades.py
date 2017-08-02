import os
import sys

a = []
with open('../data/-cascades256.txt', "r") as f:
    for line in f:
        line = line.strip();
        if len(line) == 0:
            continue
        nums = line.split(",")
        b = []
        for i in xrange(0, len(nums), 2):
            b.append((int(nums[i]), float(nums[i+1])))
        b.sort(key = lambda t : t[1])
        a.append(b)
with open('../data/casecades256_sorted', "w") as f:
    for item in a:
        for it in item:
            f.write(str(it[0]) + "\t" + str(it[1]) + "\t")
        f.write("\n")

