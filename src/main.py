import os
import sys
import re
import json
import math
import argparse
import time
import numpy as np
import networkx as nx
import tensorflow as tf
from operator import itemgetter

from env import *
from data_handler import DataHandler as dh
from metric import Metric

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def train_model(params, is_save = True):
    data_set = dh.load_casecades(os.path.join(DATA_PATH, params["data_file"]))
    var_list = [0, 0]
    def get_batch(batch_size):
        batch_x = []
        batch_y = []
        for _ in xrange(batch_size):
            y = np.zeros(params["num_node"])
            for i in xrange(var_list[1], -1, -1):
                delta = data_set[var_list[0]][var_list[1]][1] - data_set[var_list[0]][i][1]
                if delta > params["T"]:
                    break
                y[data_set[var_list[0]][i][0]] = math.exp(
                        -params["K"] * delta)
            batch_y.append(y)
            batch_x.append([data_set[var_list[0]][var_list[1]][0]])
            var_list[1] += 1
            if var_list[1] == len(data_set[var_list[0]]):
                var_list[1] = 0
                var_list[0] = (var_list[0]+ 1) % len(data_set)
        return batch_x, batch_y

    nkg = params["model"](params)
    embeddings, coefficient = nkg.Train(get_batch,
                                        epoch_num = params["iteration"])
    if is_save:
        d = {"embeddings": embeddings.tolist(), "coefficient": coefficient.tolist()}
        file_path = os.path.join(RES_PATH, "training_res_" + str(int(time.time() * 1000.0)))
        with open(file_path, "w") as f:
            f.write(json.dumps(d))
        try:
            os.symlink(file_path, os.path.join(RES_PATH, "TrainRes"))
        except OSError:
            os.remove(os.path.join(RES_PATH, "TrainRes"))
            os.symlink(file_path, os.path.join(RES_PATH, "TrainRes"))

def metric(params):
    G_truth = dh.load_ground_truth(os.path.join(DATA_PATH, params["ground_truth_file"]))
    for metric in params["metric_function"]:
        getattr(Metric, metric["func"])(G_truth, metric)


def main():
    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--operation', type = str, default = "all", help = "[all | train | metric | draw]")
    parser.add_argument('--conf', type = str, default = "default")
    parser.add_argument('--iteration', type = int, default = 10001)
    parser.add_argument('--model', type = str, default = "model")
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))
    params["iteration"] = args.iteration

    module = __import__(args.model).NodeSkipGram
    params["model"] = module

    if args.operation == "all":
        train_model(params)
        metric(params)
    elif args.operation == "train":
        train_model(params)
    elif args.operation == "metric":
        metric(params)
    elif args.operation == "draw":
        pass
    else:
        print "Not Support!"

if __name__ == "__main__":
    main()
