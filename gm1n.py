import numpy as np
import pandas as pd
from gm11 import gm11

class gm1n(gm11):
    """
    定义GM(1,N)模型
    """
    def __init__(self, sys_data):
        super(gm1n, self).__init__()
        self.data = sys_data
        pass

    def __lsm(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def loss(self):
        pass