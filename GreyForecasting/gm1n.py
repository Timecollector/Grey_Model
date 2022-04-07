import numpy as np
import pandas as pd


class gm1n(object):
    """
    定义GM(1,N)模型
    rel_data:相关因素序列
    sys_data:系统行为序列
    predict_step:预测步长
    discrete:使用离散模型还是连续模型，默认为False，即使用连续模型进行预测
    background_coff:背景值系数，默认值为0.5
    """

    def __init__(self, rel_data: pd.DataFrame, sys_data: pd.DataFrame, predict_step: int = 2,
                 discrete: bool = False, background_coff: float = 0.5):
        self.sys_data = sys_data
        self.rel_data = rel_data
        self.data_shape = self.sys_data.shape[0]-predict_step
        self.predict_step = predict_step
        self.discrete = discrete
        self.bgc = background_coff
        self.coff = None
        self.sim_data = [self.sys_data[0]]
        self.pred_data = []

    def __lsm(self):
        # 定义矩阵Y
        Y = self.sys_data.iloc[1:self.data_shape].values.reshape((self.data_shape - 1, 1))
        # 计算背景值
        cum_sys_data = np.cumsum(self.sys_data)
        Z = np.zeros((self.data_shape - 1, 1))
        for i in range(self.data_shape - 1):
            Z[i] = self.bgc * cum_sys_data.iloc[i] + (1 - self.bgc) * cum_sys_data.iloc[i + 1]
        # 计算相关序列的累加
        rel_data_cum = np.cumsum(self.rel_data[:-self.predict_step], axis=0)
        rel_data_cum = rel_data_cum.iloc[1:self.data_shape, :].values
        # 得到矩阵B
        B = np.column_stack((-Z, rel_data_cum))
        # 使用最小二乘求解系数
        self.coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))

    def fit(self):
        self.__lsm()
        sys_data = self.sys_data.copy().iloc[:self.data_shape]
        sys_data_cum = np.cumsum(sys_data).values
        rel_data_cum = np.cumsum(self.rel_data, axis=0)
        rel_data_cum = rel_data_cum.values
        # 离散的情况
        if self.discrete:
            for i in range(1,self.data_shape+self.predict_step):
                x = 0
                for j in range(rel_data_cum.shape[1]):
                    x += (self.coff[j+1]/(1+0.5*self.coff[0]))*rel_data_cum[i,j]
                y = x - (self.coff[0]/(1+0.5*self.coff[0])) * sys_data_cum[i]
                if i >= self.data_shape-1:
                    sys_data = np.append(sys_data,y[0])
                    sys_data_cum = np.cumsum(sys_data)
                self.sim_data.append(y[0])
        # 连续的情况
        else:
            temp_lt = [sys_data_cum[0]]
            for i in range(1, self.data_shape+self.predict_step):
                x = 0
                z = 0
                for j in range(rel_data_cum.shape[1]):
                    x += self.coff[j+1] * rel_data_cum[i,j]
                    z += rel_data_cum[i,j]
                y = (sys_data_cum[0]-(1/self.coff[0])*x)*np.exp(-self.coff[0]*i)+(1/self.coff[0])*x
                temp_lt.append(y[0])
            for i in range(len(temp_lt)-1):
                x = temp_lt[i+1]-temp_lt[i]
                self.sim_data.append(x)
        return self.sim_data[:-self.predict_step]

    def predict(self):
        return self.sim_data[-self.predict_step:]


    def loss(self):
        pass


if __name__ == '__main__':
    data = pd.read_excel('Power.xlsx', sheet_name='Sheet3',header=None)
    system_data = data.iloc[:, 0]
    relevent_data = data.iloc[:, 1:]

    model = gm1n(relevent_data, system_data,predict_step=3,discrete=True)
    fit_values = model.fit()
    predict_values = model.predict()
