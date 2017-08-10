import sys
import os
import re
import math
import numpy as np

from utils.env import *
from utils.data_handler import DataHandler as dh

class BatchStrategy(object):
    def __init__(self, params):
        self.data_set = dh.load_cascades(os.path.join(DATA_PATH, params["data_file"]))
        self.num_nodes = params["num_node"]
        self.K = params["K"]
        self.T = params["T"]
        if re.match(r'.*sequential.*', params["batch_func"]) is not None:
            self.n = 0
            self.m = 0
        if re.match(r'.*avg.*', params["batch_func"]):
            self.avg_delta = dh.cal_average_delta(self.data_set, self.num_nodes, self.K, self.T)


    def sequential_avg(self, batch_size):
        batch_x = []
        batch_y = []
        for _ in xrange(batch_size):
            batch_x.append([self.n])
            batch_y.append(self.avg_delta[self.n])
            self.n = (self.n + 1) % self.num_nodes
        return batch_x, batch_y


    def sequential_origin(self, batch_size):
        batch_x = []
        batch_y = []
        for _ in xrange(batch_size):
            y = np.zeros(self.num_nodes)
            for i in xrange(self.m, -1, -1):
                delta = self.data_set[self.n][self.m][1] - self.data_set[self.n][i][1]
                if delta > self.T:
                    break
                y[self.data_set[self.n][i][0]] = math.exp(
                        - self.K * delta)
            batch_y.append(y)
            batch_x.append([self.data_set[self.n][self.m][0]])
            self.m += 1
            if self.m == len(self.data_set[self.n]):
                self.m = 0
                self.n = (self.n + 1) % len(self.data_set)
        return batch_x, batch_y
