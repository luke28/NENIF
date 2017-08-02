import os
import sys
import json
import time

from env import *

class DataHandler(object):
    @staticmethod
    def cal_delta(data, is_save = False):
        ret = []
        for item in data:
            tmp = []
            for i in xrange(len(item) - 2, 0, -1):
                tmp.append((item[i][0],
                    item[i][1] - item[i-1][1]))
            tmp.append((item[0][0], 0))
            tmp.reverse()
            ret.append(tmp)
        if is_save:
            with open(os.path.join(DATA_PATH, "delta_casecades_" + str(int(time.time() * 1000.0))), "w") as f:
                f.write(json.dumps(ret))
        return ret

def main():
    data = []
    with open('../data/casecades256_sorted', "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split("\t")
            tmp = []
            for i in xrange(0, len(items), 2):
                tmp.append((int(items[i]),float(items[i+1])))
            data.append(tmp)

    ret = DataHandler.cal_delta(data, True)

if __name__ == "__main__":
    main()
