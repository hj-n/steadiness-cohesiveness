import time
import tadasets
import umap

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import pandas as pd
import csv

import json

PATH_TO_WEB = "../../web/src/json/"
PATH_TO_MEASURE = "./../fimif_measure/json/"
PATH = PATH_TO_MEASURE

json_file = 