import os
import sys
import networkx as nx
import re
import json

class DataHandler(object):
    @staticmethod
    def load_cascades(file_path):
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

    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as f:
            s = f.read()
            s = re.sub('\s',"", s)
        return json.loads(s)

    @staticmethod
    def append_to_file(file_path, s):
        with open(file_path, "a") as f:
            a.write(s)

    @staticmethod
    def load_ground_truth(file_path):
        G = nx.DiGraph()
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split(",")
                G.add_edge(int(items[0]), int(items[1]))
        return G
