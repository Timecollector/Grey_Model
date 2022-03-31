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




if __name__ == '__main__':
    data_mic = pd.read_excel('C:\\Users\\ZhangYiFan\\Desktop\\2021年中国研究生数学建模竞赛赛题\\2021年D题\\MIC.xlsx')
    data_in = pd.read_excel('C:\\Users\\ZhangYiFan\\Desktop\\2021年中国研究生数学建模竞赛赛题\\2021年D题\\Molecular_Descriptor.xlsx')
    data_out = pd.read_excel('C:\\Users\\ZhangYiFan\\Desktop\\2021年中国研究生数学建模竞赛赛题\\2021年D题\\ERα_activity.xlsx')
    data_in = data_in[data_mic['Variable'][data_mic['MIC'] > 0.1]]
    data_in = mean_process(data_in)
    data_out = mean_process(data_out.iloc[:, 1:])
    a = gery_inci(data_in, data_in)
    sns.heatmap(a)