import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cum_fuc
import opeators
import math
from sklearn.preprocessing import MinMaxScaler

'''
定义最基本的GM(1,1)模型
'''


class gm11():
    def __init__(self):
        self.data = []
        self.coff_val = []
        self.simulate_val = []
        self.predict_val = []
        self.simulate_val_all = []
        self.residual = []
        self.relative_error = []
        self.simulate_val_all = []

    def fit(self, data, predict_steps=2):
        self.data = data
        data_cum = np.cumsum(data)  # 累加
        Y = data.reshape(len(data), 1)[1:]
        data_cum_z = []
        for i in range(1, len(data_cum)):
            z = 0.5 * (data_cum[i] + data_cum[i - 1])
            data_cum_z.append(z)  # 使用均值形式
        data_cum_z = np.array(data_cum_z)
        B1 = -data_cum_z.reshape(len(data_cum_z), 1)
        B2 = np.ones(len(data_cum) - 1).reshape(len(data_cum) - 1, 1)  # 生成一个列向量，元素都是1
        B = np.hstack((B1, B2))

        # 用最小二乘求解a，b
        coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
        a, b = coff[0][0], coff[1][0]
        self.coff_val.append(a)
        self.coff_val.append(b)

        # 模拟真实值
        self.simulate_val.append(data[0])
        for i in range(2, len(data_cum) + 1):  # 从第二期开始模拟
            x = (1 - np.exp(a)) * (data[0] - b / a) * np.exp(-a * (i - 1))
            self.simulate_val.append(x)

        # 预测
        for i in range(len(data_cum) + 1, len(data_cum) + predict_steps + 1):
            x = (1 - np.exp(a)) * (data[0] - b / a) * np.exp(-a * (i - 1))
            self.predict_val.append(x)

        self.simulate_val_all = self.simulate_val + self.predict_val

        # 误差计算
        # 残差
        self.residual = data - self.simulate_val
        # 相对误差
        for i in range(len(data)):
            x = abs(self.residual[i]) / data[i]
            self.relative_error.append(x)

    # 获取拟合值
    def get_sim_val(self):
        return self.simulate_val

    # 获取系数
    def get_coff_val(self):
        return self.coff_val

    # 获取预测值
    def get_pred_val(self):
        return self.predict_val

    # 获取相对误差
    def get_error(self):
        return self.relative_error

    # 画图
    def gm_plot(self, predict_steps=2):
        plt.plot(self.simulate_val_all, label='simulate value')
        plt.plot(self.data, label='real value')
        plt.vlines(len(self.simulate_val_all) - predict_steps - 1, 0, max(self.data) + min(self.data),
                   label='predict value')
        plt.xlabel('time')
        plt.ylabel('cases')
        plt.legend()
        plt.show()


'''
定义滚动GM(1,1)模型
'''


class roll_gm11():
    def __init__(self):
        self.data = []
        self.a = []
        self.b = []
        self.simulate_val = []
        self.predict_val = []
        self.simulate_val_all = []
        self.residual = []
        self.relative_error = []
        self.simulate_val_all = []

    def fit(self, data, roll_step=4, predict_step=1):
        # 数据预处理
        data = data.reshape(1, len(data))
        self.data = [x for y in data for x in y]  # 用于画图
        data = [x for y in data for x in y]  # 降维

        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.window(roll_step, shift=1, drop_remainder=True)
        data = data.flat_map(lambda window: window.batch(roll_step))

        count = 0  # 增加一个计数器，用于记录迭代次数
        for item in data:
            item = np.array(item)  # 把列表转化为np.array
            data_cum = np.cumsum(item)  # 累加
            Y = item.reshape(len(item), 1)[1:]
            data_cum_z = []
            for i in range(1, len(data_cum)):
                z = 0.5 * (data_cum[i] + data_cum[i - 1])
                data_cum_z.append(z)  # 使用均值形式
            data_cum_z = np.array(data_cum_z)
            B1 = -data_cum_z.reshape(len(data_cum_z), 1)
            B2 = np.ones(len(data_cum) - 1).reshape(len(data_cum) - 1, 1)  # 生成一个列向量，元素都是1
            B = np.hstack((B1, B2))

            # 用最小二乘求解a，b,把每一次循环的系数保存到列表中
            coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
            self.a.append(coff[0][0])
            self.b.append(coff[1][0])

            for i in range(len(data_cum) + 1, len(data_cum) + predict_step + 1):
                x = (1 - np.exp(self.a[count])) * (item[0] - self.b[count] / self.a[count]) * np.exp(
                    -self.a[count] * (i - 1))  # 用count获取a，b
                self.predict_val.append(x)
            count += 1

        # 误差计算
        # 残差
        data_cal = self.data[roll_step:]
        for i in range(len(data_cal)):
            x = data_cal[i] - self.predict_val[i]
            self.residual.append(x)
        # 相对误差
        for i in range(len(data_cal)):
            x = abs(self.residual[i]) / data_cal[i]
            self.relative_error.append(x)

        # 获取系数

    def get_coff_val(self):
        return self.a, self.b

        # 获取预测值

    def get_pred_val(self):
        return self.predict_val

        # 获取相对误差

    def get_error(self):
        return self.relative_error

    # 画图
    def gm_plot(self, roll_step=4, predict_step=1):
        plt.plot(self.predict_val, label='predict value')
        plt.plot(self.data[roll_step:], label='real value')
        plt.vlines(len(self.predict_val) - predict_step - 1, 0, max(self.data) + min(self.data),
                   label='predict value')
        plt.xlabel('time')
        plt.ylabel('cases')
        plt.legend()
        plt.show()


'''
定义GM(1,N|sin)幂模型
当p=0，退化到GM(1,N)幂模型
当p=0，gama均为1，退化到GM(1,N)模型
'''


class pgm1ns():
    def __init__(self):
        self.data = []
        self.coff = []
        self.simulate_val = []
        self.predict_val = []
        self.simulate_val_all = []
        self.residual = []
        self.relative_error = []
        self.error = []
        self.simulate_val_all = []
        self.pre_error = []

    def fit(self, data, predict_step=2, gama=[1.7254, 0.86642], p=1.5):
        var1 = data[:, 0]
        self.data = var1
        var2 = data[:, 1:]
        var1_sim = var1[:-predict_step]
        var2_sim = var2[:-predict_step, :]
        var2_size = var2.shape[1]  # 获取一共有几个变量

        var1_sim_cum = np.cumsum(var1_sim)
        for i in range(var2_sim.shape[1]):
            var2_sim = np.hstack((var2_sim, np.cumsum(var2_sim[:, i]).reshape(var2_sim.shape[0], 1)))
        var2_sim_cum = var2_sim[:, var2_size:]  # 取出累加后的变量

        Y = var1_sim[1:]
        var1_sim_z = []
        for i in range(1, len(var1_sim_cum)):
            z = 0.5 * var1_sim_cum[i - 1] + 0.5 * var1_sim_cum[i]
            var1_sim_z.append(z)
        var1_sim_z = np.array(var1_sim_z).reshape(len(var1_sim_z), 1)

        # 求幂
        for i in range(var2_sim_cum.shape[1]):
            var2_sim_cum = np.hstack((var2_sim_cum, (var2_sim_cum[:, i] ** gama[i]).reshape(var2_sim.shape[0], 1)))
        var2_sim_cum_exp = var2_sim_cum[:, var2_size:]

        # 获取三角函数和1向量
        sin_list = []
        for i in range(2, len(var1_sim) + 1):
            x = math.sin(i * p)
            sin_list.append(x)
        sin_list = np.array(sin_list).reshape(len(sin_list), 1)
        ones_list = np.ones(len(var1_sim) - 1).reshape(len(var1_sim) - 1, 1)

        # 组成矩阵
        if p == 0:
            var1_sim_z = -var1_sim_z
            B = np.hstack((var1_sim_z, var2_sim_cum_exp[1:, :]))
            # 最小二乘计算系数
            self.coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
            self.coff = np.append(self.coff, [0])
            self.coff = np.append(self.coff, [0])

            # 计算模拟值
            for i in range(var2_size):
                var2_sim_cum_exp = np.hstack((var2_sim_cum_exp,
                                              (var2_sim_cum_exp[:, i] * self.coff[i + 1] / (
                                                      1 + 0.5 * self.coff[0])).reshape(var2_sim_cum_exp.shape[0],
                                                                                       1)))
            y = var2_sim_cum_exp[:, var2_size:]
            self.simulate_val.append(var1[0])
            for i in range(1, len(var1_sim)):
                x = sum(y[i, :]) - var1_sim_cum[i - 1] * (self.coff[0] / (1 + 0.5 * self.coff[0])) + \
                    (self.coff[-2] * math.sin(p * i)) / (1 + 0.5 * self.coff[0]) + self.coff[-1] / (
                            1 + 0.5 * self.coff[0])
                self.simulate_val.append(x)

            # 计算预测值
            for i in range(var2.shape[1]):
                var2 = np.hstack((var2, np.cumsum(var2[:, i]).reshape(var2.shape[0], 1)))
            var2_cum_all = var2[:, var2_size:]

            for i in range(var2_cum_all.shape[1]):
                var2_cum_all = np.hstack(
                    (var2_cum_all, (var2_cum_all[:, i] ** gama[i]).reshape(var2_cum_all.shape[0], 1)))
            var2_cum_all_exp = var2_cum_all[:, var2_size:]
            for i in range(var2_size):
                var2_cum_all_exp = np.hstack((var2_cum_all_exp,
                                              (var2_cum_all_exp[:, i] * self.coff[i + 1] / (
                                                      1 + 0.5 * self.coff[0])).reshape(var2.shape[0], 1)))
            y = var2_cum_all_exp[:, var2_size:]
            for i in range(len(var1_sim_cum), len(var1_sim_cum) + predict_step):
                x = sum(y[i, :]) - var1_sim_cum[i - 1] * (self.coff[0] / (1 + 0.5 * self.coff[0])) + \
                    (self.coff[-2] * math.sin(p * i)) / (1 + 0.5 * self.coff[0]) + self.coff[-1] / (
                            1 + 0.5 * self.coff[0])
                self.predict_val.append(x)
                var1_sim_cum = np.append(var1_sim_cum, var1_sim_cum[i - 1] + x)
        else:
            var1_sim_z = -var1_sim_z
            B = np.hstack((var1_sim_z, var2_sim_cum_exp[1:, :], sin_list, ones_list))
            # 最小二乘计算系数
            self.coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))

            # 计算模拟值
            for i in range(var2_size):
                var2_sim_cum_exp = np.hstack((var2_sim_cum_exp,
                                              (var2_sim_cum_exp[:, i] * self.coff[i + 1] / (
                                                          1 + 0.5 * self.coff[0])).reshape(var2_sim_cum_exp.shape[0],
                                                                                           1)))
            y = var2_sim_cum_exp[:, var2_size:]
            self.simulate_val.append(var1[0])
            for i in range(1, len(var1_sim)):
                x = sum(y[i, :]) - var1_sim_cum[i - 1] * (self.coff[0] / (1 + 0.5 * self.coff[0])) + \
                    (self.coff[-2] * math.sin(p * i)) / (1 + 0.5 * self.coff[0]) + self.coff[-1] / (
                                1 + 0.5 * self.coff[0])
                self.simulate_val.append(x)

            # 计算预测值
            for i in range(var2.shape[1]):
                var2 = np.hstack((var2, np.cumsum(var2[:, i]).reshape(var2.shape[0], 1)))
            var2_cum_all = var2[:, var2_size:]

            for i in range(var2_cum_all.shape[1]):
                var2_cum_all = np.hstack(
                    (var2_cum_all, (var2_cum_all[:, i] ** gama[i]).reshape(var2_cum_all.shape[0], 1)))
            var2_cum_all_exp = var2_cum_all[:, var2_size:]
            for i in range(var2_size):
                var2_cum_all_exp = np.hstack((var2_cum_all_exp,
                                              (var2_cum_all_exp[:, i] * self.coff[i + 1] / (
                                                          1 + 0.5 * self.coff[0])).reshape(var2.shape[0], 1)))
            y = var2_cum_all_exp[:, var2_size:]
            for i in range(len(var1_sim_cum), len(var1_sim_cum) + predict_step):
                x = sum(y[i, :]) - var1_sim_cum[i - 1] * (self.coff[0] / (1 + 0.5 * self.coff[0])) + \
                    (self.coff[-2] * math.sin(p * i)) / (1 + 0.5 * self.coff[0]) + self.coff[-1] / (
                                1 + 0.5 * self.coff[0])
                self.predict_val.append(x)
                var1_sim_cum = np.append(var1_sim_cum, var1_sim_cum[i - 1] + x)

        # 整合
        self.simulate_val_all = self.simulate_val + self.predict_val

        # 误差计算
        # 残差
        self.residual = var1_sim - self.simulate_val
        # 相对误差
        for i in range(len(var1_sim)):
            x = abs(self.residual[i]) / var1_sim[i]
            self.relative_error.append(x)

        for i in range(1, predict_step + 1):
            x = abs(var1[-i] - self.simulate_val_all[-i]) / self.simulate_val_all[-i]
            self.pre_error.append(x)

        for i in range(len(var1_sim)):
            x = abs(var1_sim[i] - self.simulate_val[i]) / var1_sim[i]
            self.error.append(x)

    def get_coff_val(self):
        return self.coff

    # 获取预测值
    def get_pred_val(self):
        return self.predict_val

    # 获取相对误差
    def get_error(self):
        return self.relative_error

    # 获取预测误差
    def get_pre_mean_error(self):
        return sum(self.pre_error) / len(self.pre_error)

    def get_pre_error(self):
        return self.pre_error

    # 获取总体平均误差
    def get_mean_error(self):
        return sum(self.error) / (len(self.simulate_val) - 1)

    # 获取全部拟合值
    def get_all_sim_val(self):
        return self.simulate_val_all

    # 画图
    def gm_plot(self, predict_step=2):
        plt.plot(self.data, label='Real Value', marker='o')
        plt.plot(self.simulate_val_all, label='Simulate Value', marker='+')
        plt.vlines(len(self.simulate_val_all) - predict_step - 1, min(self.data), max(self.data) - min(self.data) / 2,
                   label='predict')
        plt.xlabel('time')
        plt.ylabel('cases')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    data = pd.read_excel('Power.xlsx',header=None)
    data = np.array(data)
    a = gm11()
    a.fit(data=data)
    print(a.simulate_val_all)
