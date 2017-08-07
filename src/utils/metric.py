import os
import sys
import time
import networkx as nx
import json
import numpy as np
from operator import itemgetter

from env import *
from data_handler import DataHandler as dh

class Metric(object):
    @staticmethod
    def cal_metric(TP, FP, TN, FN):
	res = {}
	res["acc"] = float(TP + TN) / float(TP + FP + FN + TN)
	res["precision"] = float(TP) / float(TP + FP) if TP + FP > 0 else 1.0
	res["recall"] = float(TP) / float(TP + FN) if TP + FN > 0 else 1.0
	res["F1"] = 1.0 / (1.0 / res["recall"] + 1.0 / res["precision"])
	return res

    @staticmethod
    def knn(G_truth, params):
        embeddings = dh.load_json_file(os.path.join(RES_PATH, "TrainRes"))["embeddings"]
        embeddings = np.array(embeddings)
        TP = FP = TN = FN = 0
        tmp = []
        for i in xrange(len(embeddings)):
            tmp.append([])
            for j in xrange(len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                tmp[-1].append((dist, i, j))
            tmp[-1].sort(key = itemgetter(0))
        for item in tmp:
            for it in item[:params["K"]]:
                if it[1] in G_truth and it[2] in G_truth[it[1]]:
                    TP += 1
                else:
                    FP += 1
            for it in item[params["K"]:]:
                if it[1] in G_truth and it[2] in G_truth[it[1]]:
                    FN += 1
                else:
                    TN += 1

        res = Metric.cal_metric(TP, FP, TN, FN)
        print "knn metric:"
        print res
        return res

    @staticmethod
    def coefficient(G_truth, params):
        coefficient = dh.load_json_file(os.path.join(RES_PATH, "TrainRes"))["coefficient"]
        TP = FP = TN = FN = 0
        for i in xrange(len(coefficient)):
            for j in xrange(len(coefficient[i])):
                if abs(coefficient[i][j]) > params["threshold"]:
                    if i in G_truth and j in G_truth[i]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if i in G_truth and j in G_truth[i]:
                        FN += 1
                    else:
                        TN += 1
        res = Metric.cal_metric(TP, FP, TN, FN)
        print "coefficient metric:"
        print res
        return res
