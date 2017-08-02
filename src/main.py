import os
import sys
import json
import math
import re
import argparse
import time
import numpy as np

from env import *
from model import NodeSkipGram

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_data(file_path):
    data_set = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split("\t")
            tmp = []
            for i in xrange(0, len(items), 2):
                tmp.append((int(items[i]),float(items[i+1])))
            data_set.append(tmp)
    return data_set

def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)


def init():
    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conf', type = str, default = "default")
    parser.add_argument('--iteration', type = int, default = 10001)
    args = parser.parse_args()
    params = load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))
    params["iteration"] = args.iteration
    n = 0
    m = 0
    data_set = load_data(os.path.join(DATA_PATH, params["data_file"]))
    return data_set, params, n, m

def main():
    data_set, params, n, m = init()
    var_list = [n, m]
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
    nkg = NodeSkipGram(params)
    embeddings, coefficient = nkg.Train(get_batch, epoch_num = params["iteration"])
    d = {"embeddings": embeddings.tolist(), "coefficient": coefficient.tolist()}
    file_path = os.path.join(RES_PATH, "training_res_" + str(int(time.time() * 1000.0)))
    with open(file_path, "w") as f:
        f.write(json.dumps(d))
    os.symlink(file_path, os.path.join(RES_PATH, "TrainRes"))

if __name__ == "__main__":
    main()
