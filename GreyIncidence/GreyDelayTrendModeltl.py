import numpy as np
import pandas as pd
import math


def mean_process(df):
    cn = df.shape[1]
    dfc = df.copy()
    for i in range(cn):
        dfc.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean()) / np.std(df.iloc[:, i])
    return dfc


def tdiff(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    n = df.shape[0]
    if t >= n:
        raise ValueError('t={}大于或等于序列长度{}'.format(t, n))
    diff_mat = np.zeros((n - t, 1))
    for i in range(0, n - t):
        diff_mat[i] = (df.iloc[i + t, 0] - df.iloc[i, 0])/t
    return pd.DataFrame(diff_mat)


def sgn(a, b):
    if a * b == 0:
        return 1
    else:
        return a * b / abs(a * b)


def gdtep_t(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    t_range = df_1.shape[0] - 1
    mat = np.full((t_range, t_range), fill_value=np.nan, dtype=np.float)
    for t in range(1, t_range+1):
        df1_diff = tdiff(df_1, t=t)
        df2_diff = tdiff(df_2, t=t)
        for i in range(df1_diff.shape[0]):
            x = sgn(df1_diff.iloc[i,0], df2_diff.iloc[i, 0]) * \
                (1 - math.sin(abs(math.atan(df1_diff.iloc[i,0])-math.atan(df2_diff.iloc[i,0]))))
            mat[t-1, i] = x
    mat = pd.DataFrame(mat)
    mat = pd.concat([mat, pd.DataFrame(mat.mean(axis=1))],axis=1)
    return mat



if __name__ == '__main__':
    data = pd.read_excel('C:\\Users\\ZhangYiFan\\Desktop\\资料\\我的坚果云\\08.数据集\\南京大气污染物\\南京大气污染影响因素整理.xlsx',
                         sheet_name='Sheet4',header=None)
    data1 = mean_process(data.iloc[:, 0:1])
    data2 = mean_process(data.iloc[:, 1:2])
    a = gdtep_t(data1, data2)
