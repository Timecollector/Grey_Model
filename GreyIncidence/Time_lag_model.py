import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 定义初值化函数 df -> df
def initi_process(df):
    cm = df.shape[1]
    dfc = df.copy()
    for i in range(cm):
        dfc.iloc[:,i] = df.iloc[:,i]/df.iloc[0,i]
    return dfc

# 定义归一化函数 df -> df
def maxmin_process(df):
    cn = df.shape[1]
    dfc = df.copy()
    for i in range(cn):
        dfc.iloc[:,i] = (df.iloc[:,i] - min(df.iloc[:,i]))/(max(df.iloc[:,i])-min(df.iloc[:,i]))
    return dfc

# 定义均值化函数 df -> df
def mean_process(df):
    cn = df.shape[1]
    dfc = df.copy()
    for i in range(cn):
        dfc.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean()) / np.std(df.iloc[:, i])
    return dfc


# 定义斜率变化计算函数 df -> df
def diff(df):
    l = df.shape[0]
    cn = df.shape[1]
    dfc = df.copy()
    for i in range(cn):
        for j in range(l-1):
            dfc.iloc[l-1-j,i] = df.iloc[l-j-1,i]-df.iloc[l-j-2,i]
    return dfc


# 定义sgn函数 float -> float
def sgn(a,b):
    if a*b == 0:
        return 1
    else:
        return a*b / abs(a*b)

def modify_func(a,b):
    if a*b >= 0:
        return 1
    else:
        return -1


# 定义在某个时滞期下的时滞关联函数 df -> array
def time_lag_inc(input,output,lag=1,process_func=initi_process):
    inshr = input.shape[0]
    # 对输入输出处理，数据处理以及获得斜率
    input_diff = diff(process_func(input))
    output_diff = diff(process_func(output))
    gi_mat = np.zeros((input.shape[1],output.shape[1]))
    for i in range(input.shape[1]):
        for j in range(output.shape[1]):
            # 构建子矩阵，存储点与点之间的关联度
            sub_gi_mat = np.zeros((1,input.shape[0]))
            for m in range(inshr-1-lag):
                sub_gi_mat[0,inshr-m-1] = sgn(output_diff.iloc[inshr-1-m,j],input_diff.iloc[inshr-1-m-lag,i])*\
                                      (1+abs(output_diff.iloc[inshr-1-m,j])+abs(input_diff.iloc[inshr-1-m-lag,i]))/\
                                      (1+abs(output_diff.iloc[inshr-1-m,j])+abs(input_diff.iloc[inshr-1-m-lag,i])+
                                       0.5*abs(output_diff.iloc[inshr-1-m,j]-
                                        # modify_func(output_diff.iloc[inshr-1-m,j],input_diff.iloc[inshr-1-m-lag,i])*
                                        input_diff.iloc[inshr-1-m-lag,i]))
            # 将点关联度求和得到灰色关联度
            gi_mat[i,j] = sub_gi_mat.sum()/(inshr-lag-1)
    return gi_mat

'''
做个修改版
'''
# def time_lag_inc_improved(input,output,lag=1,process_func=initi_process):
#     inshr = input.shape[0]
#     coeff = {0: 1, 1: 0.618034, 2: 0.543689, 3: 0.51879, 4: 0.50866}
#     input_lag_sum =


# 定义寻找时滞期的函数 array -> array
def find_time_lag(input,output,max_lag=3,threshold=0.5):
    # 定义一个元素都为-1的矩阵存储时滞期 若最后元素仍然为-1，则代表两个元素间没有时滞关系
    tl_mat = np.zeros((input.shape[1],output.shape[1])) + 9999
    # 定义一个三维矩阵存储时滞关联系数
    tlgi_mat = np.zeros((max_lag+1,input.shape[1],output.shape[1]))
    for lag_num in range(max_lag+1):
        tlgi_mat[lag_num] = time_lag_inc(input,output,lag=lag_num)
    for i in range(input.shape[1]):
        for j in range(output.shape[1]):
            # 在矩阵一个个位置搜索，如果大于阈值，就停止搜索
            for k in range(max_lag+1):
                if abs(tlgi_mat[k][i][j]) > threshold:
                    tl_mat[i][j] = k
                    break
    return tl_mat

if __name__ == '__main__':
    data = pd.read_excel('C:\\Users\\ZhangYiFan\\Desktop\\资料\\我的坚果云\\08.数据集\\南京大气污染物\\南京大气污染影响因素整理 - 副本.xlsx')
    data = data.drop(labels='数据来源：南京市统计局、江苏省统计局、南京市气象局',axis=1)
    data_input = data.iloc[14:,1:9]
    data_output = data.iloc[14:,9:]
    inc_mat = time_lag_inc(data_input,data_output,3)
    time_lag_matrix = find_time_lag(data_input,data_output,max_lag=3,threshold=0.5)