__author__ = '张逸凡'

import numpy as np
import pandas as pd


class gm11(object):
    """
    经典GM(1,1)模型

    使用方法：
    1.实例化类     model = gm11(data,predstep=2)
    2.训练模型     model.fit()
    3.查看拟合误差  model.MSE()
    4.预测        model.predict()

    Ps:背景值系数bg_coff接收的是一个列表（主要为了后面新模型的构建），默认值为一个空列表，此时背景值系数默认全部为0.5
    """
    def __init__(self, sys_data: pd.DataFrame, predstep: int = 2, bg_coff: list = []):
        if not isinstance(sys_data, pd.DataFrame):
            raise ImportError('请使用Dataframe格式载入数据')
        if not isinstance(predstep,int):
            raise ValueError('predstep必须是整数')
        # if bg_coff > 1 or bg_coff < 0:
        #     raise ValueError('background_coff必须在[0,1]之间')
        self.data = np.array(sys_data.iloc[:, 0].values)
        self.data_shape = self.data.shape[0]
        self.data = self.data.reshape((self.data_shape,1))
        self.coff = []
        self.sim_values = np.zeros((self.data_shape,1))
        self.predstep = predstep
        self.pred_values = np.zeros((self.predstep,1))
        self.error = []
        self.bg_coff = bg_coff
        self.rel_errors = []

    def __lsm(self):
        data_cum = np.cumsum(self.data)
        Y = self.data[1:]
        # 计算背景值
        background_values = np.zeros((self.data_shape - 1, 1)).reshape((self.data_shape - 1, 1))
        if self.bg_coff == []:
            for i in range(1,self.data_shape):
                background_values[i-1][0] = 0.5 * data_cum[i] + (1-0.5) * data_cum[i-1]
        else:
            for i in range(1,self.data_shape):
                background_values[i-1][0] = (1-self.bg_coff[i-1]) * data_cum[i] + self.bg_coff[i-1] * data_cum[i-1]
        one_array = np.ones((self.data_shape-1,1)).reshape((self.data_shape-1,1))
        background_values = -background_values
        B = np.hstack((background_values,one_array))
        # 用最小二乘求解参数
        self.coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))

    def fit(self) -> np.array:
        self.__lsm()
        a = self.coff[0][0]
        b = self.coff[1][0]
        # 求解拟合值
        self.sim_values[0] = self.data[0][0]
        for i in range(2,self.data_shape+1):
            self.sim_values[i-1] = (1 - np.exp(a)) * (self.data[0][0] - b / a) * np.exp(-a * (i - 1))
        self.sim_values = self.sim_values.reshape((1,self.data_shape))[0]
        return self.sim_values

    def predict(self) -> np.array:
        self.__lsm()
        a = self.coff[0][0]
        b = self.coff[1][0]
        # 求解预测值
        for i in range(self.data_shape+1,self.data_shape+1+self.predstep):
            self.pred_values[i - 1 - self.data_shape][0] = (1 - np.exp(a)) * (self.data[0][0] - b / a) * np.exp(-a * (i - 1))
        self.pred_values = self.pred_values.reshape((1,self.predstep))[0]
        return self.pred_values

    def loss(self) -> list:
        for i in range(self.data_shape):
            self.error.append(abs(self.sim_values[i]-self.data[i][0])/self.data[i][0])
        return sum(self.error)/len(self.error)

    def errors(self):
        for i in range(self.data_shape):
            self.rel_errors.append(self.data[i][0]-self.sim_values[i])
        return self.rel_errors



if __name__ == '__main__':
    data = pd.read_excel('test.xlsx',header=None)
    a = gm11(data)
    print('GM(1,1)的拟合值是： ', a.fit())
    print(f'GM(1,1)的{a.predstep}步预测值是： ', a.predict())
    print('GM(1,1)的预测误差是： ', a.loss())
    print(a.errors())