import sys
import os
import math
import numpy as np
import networkx as nx

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FILE_PATH, '../src'))
from utils.data_handler import DataHandler as dh
from utils.metric import Metric

data_set = dh.load_cascades(os.path.join(FILE_PATH, r"../data/cascades256_sorted"))
n = 256
res = np.zeros((n, n), dtype = float)
cnt = np.zeros(n, dtype = float)
for data in data_set:
    for i in xrange(len(data)):
        for j in xrange(i, len(data)):
            res[data[i][0]][data[j][0]] += math.exp(float(sys.argv[1]) * abs(data[i][1] - data[j][1]))
        cnt[data[i][0]] += 1.0
G = dh.load_ground_truth(os.path.join(FILE_PATH,r"../data/network256"))
avg = 0.0
avg_have = 0.0
avg_not = 0.0
for i in xrange(n):
    for j in xrange(n):
        ans = res[i][j] / cnt[i]
        avg += ans
        if j in G[i]:
            avg_have += ans
        else:
            avg_not += ans

avg /= float(n * n)
avg_have /= float(G.number_of_edges())
avg_not /= float(n * n - G.number_of_edges())
print "total average: " + str(avg)
print "average with edges: " + str(avg_have)
print "average without edges: " + str(avg_not)

precision = {}
recall = {}
for k in xrange(1, 20):
    TP = FP = TN = FN = 0
    for i in xrange(n):
        lst = []
        for j in xrange(n):
            lst.append((res[i][j], j))
        lst.sort(reverse = True)
        for j in xrange(k):
            if lst[j][1] in G[i]:
                TP += 1
            else:
                FP += 1

    FN = G.number_of_edges() - TP
    TN = n * n - FN - FP - TP
    r = Metric.cal_metric(TP, FP, TN, FN)
    precision[k] = r["precision"]
    recall[k] = r["recall"]
Metric.draw_pr(precision, recall)
