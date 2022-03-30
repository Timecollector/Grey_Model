import numpy as np
import pandas as pd
import math
from gm1n import gm1n
import time

class pgm1nsin(object):
    """
    定义GM(1,N|sin)幂模型

    参数设置:\n
    sys_data:系统行为序列\n
    rel_p_data:相关因素序列（指数因素）\n
    rel_s_data:相关因素序列（周期因素）\n
    predict_step:预测步长，会截取上面三组数据的最后predict_step行数据作为预测\n
    gamma:幂指数\n
    p:周期系数\n
    使用方法：\n
    1.实例化对象 model = pgm1nsin(sys_data=sys_data,rel_p_data=rel_p_data,rel_s_data=rel_s_data)\n
    2.训练模型  model.fit()\n
    3.进行预测  model.predict
    """

    def __init__(self, sys_data: pd.DataFrame, rel_p_data: pd.DataFrame, rel_s_data: pd.DataFrame,
                 predict_step: int = 2, gamma: list = [1.7254], p: list = [1.5]):
        self.sys_data = sys_data
        self.rel_p_data = rel_p_data
        self.rel_s_data = rel_s_data
        if self.sys_data.shape[0] != self.rel_p_data.shape[0] or self.rel_p_data.shape[0] != self.rel_s_data.shape[0]:
            raise Warning('系统行为序列与相关因素序列长度不同，可能导致程序报错或误差计算函数无法调用')

        self.data_shape = self.sys_data.shape[0]
        # 变量维度设置
        if len(self.rel_p_data.shape) == 1:
            self.rel_p_data_shape = 1
        else:
            self.rel_p_data_shape = self.rel_p_data.shape[1]
        if len(self.rel_s_data.shape) == 1:
            self.rel_s_data_shape = 1
        else:
            self.rel_s_data_shape = self.rel_s_data.shape[1]

        if self.rel_p_data_shape != len(gamma):
            raise IndexError('传入的幂指数列表与幂指数序列长度不匹配')
        if self.rel_s_data_shape != len(p):
            raise IndexError('传入的周期数列表与周期数序列长度不匹配')

        self.gamma = gamma
        self.p = p
        self.coff = []
        self.__rel_p_data_sum = []
        self.__rel_s_data_sum = []
        self.__power_values = []
        self.__sin_values = []
        self.sim_values = []
        self.error = 10
        self.predict_step = predict_step
        self.__sum_power_values = []
        self.__sum_sin_values = []
        self.predict_values = []
        self.predict_error = 10
        self.errors = []
        self.pred_errors = []

    def __lsm(self) -> np.array:
        Y = self.sys_data.iloc[1:self.data_shape-self.predict_step].values.reshape((self.data_shape-1-self.predict_step,1))
        # 进行累加生成
        sys_data_sum = np.cumsum(self.sys_data).values.reshape((self.data_shape,1))[:-self.predict_step]
        self.__rel_p_data_sum = np.cumsum(self.rel_p_data,axis=0).values.reshape((self.data_shape,self.rel_p_data_shape))
        self.__rel_s_data_sum = np.cumsum(self.rel_s_data,axis=0).values.reshape((self.data_shape, self.rel_s_data_shape))
        # 计算背景值
        bg_values = np.zeros((self.data_shape-1-self.predict_step,1))
        for i in range(1,self.data_shape-self.predict_step):
            bg_values[i-1] = 0.5*sys_data_sum[i-1] + 0.5*sys_data_sum[i]
        # 计算幂指数序列
        self.__power_values = np.zeros((self.data_shape-1,self.rel_p_data_shape))
        for i in range(self.rel_p_data_shape):
            for j in range(1,self.data_shape):
                self.__power_values[j-1,i] = self.__rel_p_data_sum[j,i] ** self.gamma[i]
        # 计算周期系数序列
        self.__sin_values = np.zeros((self.data_shape-1,self.rel_s_data_shape))
        for i in range(self.rel_s_data_shape):
            for j in range(1,self.data_shape):
                self.__sin_values[j-1,i] = self.__rel_s_data_sum[j,i] * math.sin(self.p[i] * (j+1))
        # 合成矩阵
        ones_values = np.ones((self.data_shape-1-self.predict_step,1))
        B = np.column_stack((bg_values,self.__power_values[:-self.predict_step],self.__sin_values[:-self.predict_step],ones_values))
        self.coff = np.matmul(np.linalg.inv(np.matmul(B.T,B)),np.matmul(B.T,Y))

    def fit(self) -> list:
        self.__lsm()
        print(self.coff)
        # 计算幂指数累加值
        power_values_copy = self.__power_values.copy()
        for i in range(self.rel_p_data_shape):
            power_values_copy[:,i] = self.__power_values[:,i] * self.coff[i+1]
        self.__sum_power_values = np.sum(power_values_copy, axis=1)
        # 计算周期数累加值
        sin_values_copy = self.__sin_values.copy()
        for i in range(self.rel_s_data_shape):
            sin_values_copy[:,i] = self.__sin_values[:,i] * self.coff[i+1+self.rel_p_data_shape]
        self.__sum_sin_values = np.sum(sin_values_copy, axis=1)
        # 计算拟合值
        self.sim_values = [self.sys_data[0]]
        for i in range(self.data_shape-1-self.predict_step):
            x = self.__sum_power_values[i] / (1 + 0.5 * self.coff[0]) + self.__sum_sin_values[i] / (1 + 0.5 * self.coff[0]) + self.coff[-1] / \
                (1+0.5*self.coff[0]) - (self.coff[0] * np.cumsum(self.sys_data).values[i]) / (1+0.5*self.coff[0])
            self.sim_values.append(x[0])
        ed = time.time()
        return self.sim_values

    def predict(self) -> list:
        sys_data_c = self.sys_data.copy().values[:-self.predict_step]
        sum_sys_data = np.cumsum(sys_data_c)
        for i in range(self.data_shape-1-self.predict_step, self.data_shape-1):
            x = self.__sum_power_values[i] / (1 + 0.5 * self.coff[0]) + self.__sum_sin_values[i] / (1 + 0.5 * self.coff[0]) + self.coff[-1] / \
                (1+0.5*self.coff[0]) - (self.coff[0] * sum_sys_data[i]) / (1+0.5*self.coff[0])
            sys_data_c = np.hstack((sys_data_c, x[0]))
            sum_sys_data = np.cumsum(sys_data_c)
            self.predict_values.append(x[0])
        return self.predict_values

    def loss(self) -> float:
        for i in range(self.data_shape-self.predict_step):
            x = abs(self.sys_data[i]-self.sim_values[i])/abs(self.sys_data[i])
            self.errors.append(x)
        self.error = sum(self.errors)/len(self.errors)
        return self.error

    def pred_loss(self) -> float:
        for i in range(self.data_shape-self.predict_step,self.data_shape):
            x = abs(self.sys_data[i]-self.predict_values[i-self.data_shape+self.predict_step])/abs(self.sys_data[i])
            self.pred_errors.append(x)
        self.predict_error = sum(self.pred_errors)/len(self.pred_errors)
        return self.predict_error


if __name__ == '__main__':
    data = pd.read_excel('Power.xlsx',header=None)
    sys_data = data.iloc[:,0]
    rel_p_data = data.iloc[:,1:3]
    rel_s_data = data.iloc[:,1:3]
    """
    gamma=[2.55350441,3.], p=[1.53850847,3.]
    [2.51799181 2.97273414 2.99706826 1.53955627]
    数据增加：
    gamma=[2.71685087,3.], p=[1.43884017,1.66407522]
    [0.01       0.36516733 1.07022243 1.56358273]
    """
    a = pgm1nsin(sys_data=sys_data, rel_p_data=rel_p_data, rel_s_data=rel_s_data, gamma=[0.01,0.36516733], p=[1.07022243,1.56358273], predict_step=10)
    values = a.fit()
    pred_values = a.predict()
    print(a.coff)
    print(a.loss())
    print(a.pred_loss())

    values.extend(pred_values)
    loss = a.errors
    loss.extend(a.pred_errors)
    print(loss)

    columns = ['sim_values','error']
    values_c = pd.DataFrame(np.hstack((np.array(values).reshape((len(values),1)),np.array(loss).reshape((len(values),1)))),columns=columns)

    with pd.ExcelWriter('sim values.xlsx') as writer:
        values_c.to_excel(writer)

    import matplotlib.pyplot as plt

    x = ['14/1', '14/2', '14/3', '14/4', '15/1', '15/2', '15/3', '15/4', '16/1', '16/2', '16/3',
         '16/4', '17/1', '17/2', '17/3', '17/4', '18/1', '18/2', '18/3', '18/4', '19/1', '19/2',
         '19/3', '19/4', '20/1', '20/2', '20/3', '20/4']
    plt.plot(x, sys_data, label='Real values', marker='^', color='r')
    plt.plot(x, values, label='GM(1,N|sin) power model', marker='o', linestyle='--',
             color='cornflowerblue')
    # axs1[0,0].plot(data_time,b.get_all_sim_val(), label='GM(1,N)模型',marker='+',linestyle='--',color='lightgreen')
    plt.legend(loc='upper left', fontsize=15)
    # plt.grid(axis='y',ls='--')
    # plt.axvline(len(values) - 2.5, ls='--', color='black')
    plt.axvline(len(values) - 8.5, ls='--', color='black')
    # plt.vlines(len(b.get_all_sim_val())-1.5, min(b.get_all_sim_val()), max(b.get_all_sim_val()),linestyles='--')
    plt.annotate('Out-of-sample predictions', xy=(20, 1365), xytext=(20, 1365), fontsize=15)
    # plt.annotate('Validation', xy=(7.8, 1365), xytext=(7.8, 1365))
    # plt.annotate('←Simulation', xy=(17.5, 1365), xytext=(17.5, 1365))
    plt.xlabel('Time')
    plt.ylabel('Electricity consumption')
    plt.show()