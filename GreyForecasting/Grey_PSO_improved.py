import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from Grey_model_v2 import pgm1ns
from sko.PSO import PSO

data_train = pd.read_excel('Power.xlsx').values[:,:]
data_test = pd.read_excel('Power.xlsx').values

# #定义函数，返回值为拟合误差与验证集的预测误差平均值，使用的数据集为训练集
# def func(x):
#     x_p1, x_p2= x
#     a = pgm1ns()
#     a.fit(data_train,gama=[x_p1,x_p2],p=0)
#     return a.get_mean_error()#+0.5*a.get_pre_mean_error()
#
# #定义函数，返回值为预测误差
# def func1(x):
#     x_p1, x_p2= x
#     b = pgm1ns()
#     b.fit(data_test,gama=[x_p1,x_p2],p=0)
#     return b.get_pre_mean_error()
#
#
# error = 0.1
# while error>0.03:
#     pso = PSO(func=func,n_dim=2,lb=[0.01,0.01],ub=[3,3],max_iter=200)
#     pso.run()
#     print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
#     coff = pso.gbest_x.tolist()
#     error = abs(func1(coff))
#     print(error)


#[0.12179844,1.30965172]
model = pgm1ns()
model.fit(data_test,gama=[0.12179844,1.30965172],p=0)#[0.40485692,3.]
print('模型的平均拟合误差是： ',model.get_mean_error())
print('模型的平均预测误差是： ',model.get_pre_error())
# print('模型的参数是： ',model.get_coff_val())
print('模型的拟合/预测值是： ',model.get_all_sim_val())
print('模型的拟合相对误差是： ',model.get_error())
print('模型的预测相对误差是： ',model.get_pre_error())
print(model.coff)

values = model.get_all_sim_val()
errors = model.get_error()
errors.extend(model.get_pre_error())
columns = ['sim_values','error']
values_c = pd.DataFrame(np.hstack((np.array(values).reshape((len(values),1)),np.array(errors).reshape((len(values),1)))),columns=columns)
with pd.ExcelWriter('sim inf.xlsx') as writer:
    values_c.to_excel(writer)

model.gm_plot()

