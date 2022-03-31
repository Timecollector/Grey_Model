import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#定义均值化函数
def mean_process(df):
    cn = df.shape[1]
    dfc = df.copy()
    for i in range(cn):
        dfc.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean()) / np.std(df.iloc[:, i])
    return dfc

#定义灰色关联度
def gery_inci(var1,var2,alpha=0.8):
    shape1 = var1.shape[0]
    shape2 = var1.shape[1]
    lt_max = []
    lt_min = []
    for index1 in range(shape2):
        diff_lt = []
        for index2 in range(index1+1):
            for col in range(shape1):
                diff_lt.append(abs(var1.iloc[col,index1]-var2.iloc[col,index2]))
        lt_max.append(max(diff_lt))
        lt_min.append(min(diff_lt))
    inc_mat = np.zeros((shape2,shape2))
    for i in range(shape2):
        for m in range(i+1):
            inc_lt = []
            for j in range(shape1):
                inc_lt.append((lt_min[i]+alpha*lt_max[i])/(abs(var2.iloc[j,m]-var1.iloc[j,i])+alpha*lt_max[i]))
            inc = sum(inc_lt)/len(inc_lt)
            print(inc)
            inc_mat[i,m] = inc
    return inc_mat
