import pandas as pd
import numpy as np
from env import *

class DataSet(object):
    def __init__(self):
        self.train = pd.read_csv(TRAINSET_CSV)
        self.val = pd.read_csv(VALSET_CSV)
        self.test = pd.read_csv(TESTSET_CSV)
