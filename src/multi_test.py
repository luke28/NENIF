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

from utils.env import *
from utils.data_handler import DataHandler as dh
from utils.metric import Metric
from single_test import train_model, metric
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def dfs(lst, single_params, f, dep = 0):
    if dep == len(lst):
        train_model(single_params)
        res = metric(single_params)
        f.write(str(single_params) + "\n")
        f.write(str(res) + "\n")
        return
    if "range" in lst[dep][1]:
        for it in lst[dep][1]["range"]:
            single_params[lst[dep][0]] = it
            dfs(lst, single_params, f, dep + 1)
    else:
        item = lst[dep][1]
        it = item["start"]
        while float(item["end"] - it) > sys.float_info.epsilon:
            single_params[lst[dep][0]] = it
            dfs(lst, single_params, f, dep + 1)
            it += item["step"]

def main():
    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conf', type = str, default = "default")
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(MULTI_CONF_PATH, args.conf + ".json"))

    out_path = os.path.join(RES_PATH, "multi_res_" + str(int(time.time() * 1000.0)))
    single_params = {}
    for item in params.items():
        if item[0] == "models":
            continue
        single_params[item[0]] = item[1]
    for m in params["models"]:
        single_params["model"] = __import__(
                m["model_name"]).NodeSkipGram
        single_params["iteration"] = m["iteration"]
        tmp = [item for item in m.items() if item[0] != "model_name" and item[0] != "iteration"]
        with open(out_path, "a") as f:
            dfs(tmp, single_params, f)

    try:
        os.symlink(out_path, os.path.join(RES_PATH, "MultiRes"))
    except OSError:
        os.remove(os.path.join(RES_PATH, "MultiRes"))
        os.symlink(out_path, os.path.join(RES_PATH, "MultiRes"))

if __name__ == "__main__":
    main()
