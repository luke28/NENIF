import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(FILE_PATH, "../..")
DATA_PATH = os.path.join(ROOT_PATH, "data")
CONF_PATH = os.path.join(ROOT_PATH, "conf")
SINGLE_CONF_PATH = os.path.join(CONF_PATH, "single")
MULTI_CONF_PATH = os.path.join(CONF_PATH, "multi")
RES_PATH = os.path.join(ROOT_PATH, "res")
