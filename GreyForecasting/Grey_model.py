import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cum_fuc
import opeators
import math
from sklearn.preprocessing import MinMaxScaler

'''
目前定义了两个函数，一个是grey_model，也就是传统的GM(1,1)的均值形式
另一个是基于均值形式的滚动模型roll_gm
'''

data = np.array([20556,20083,19801,21098,20147,22689])
data1 = np.array([477,587,663,744,843,945])
data2 = np.array([1423,1790,2144,2761,3388,4169])
time = np.arange(len(data))

#GM(1,1)
def grey_model(data,time,predict_step=3):
    data_cum = np.cumsum(data)#累加
    Y = data.reshape(len(data),1)[1:]
    data_cum_z = []
    for i in range(1,len(data_cum)):
        z = 0.5* (data_cum[i]+data_cum[i-1])
        data_cum_z.append(z)#使用均值形式
    data_cum_z = np.array(data_cum_z)
    B1 = -data_cum_z.reshape(len(data_cum_z),1)
    B2 = np.ones(len(data_cum)-1).reshape(len(data_cum)-1,1)#生成一个列向量，元素都是1
    B = np.hstack((B1,B2))

    #用最小二乘求解a，b
    coff = np.matmul(np.linalg.inv(np.matmul(B.T,B)),np.matmul(B.T,Y))
    a, b = coff[0][0], coff[1][0]
    print('a={0},b={1}'.format(a,b))

    #模拟真实值
    simulate_val = []
    simulate_val.append(data[0])
    for i in range(2,len(data_cum)+1):#从第二期开始模拟
        x = (1-np.exp(a))*(data[0]-b/a)*np.exp(-a*(i-1))
        simulate_val.append(x)
    print('模拟值是： {0}'.format(simulate_val))

    #预测
    predict_val = []
    for i in range(len(data_cum)+1,len(data_cum)+predict_step+1):
        x = (1 - np.exp(a)) * (data[0] - b / a) * np.exp(-a * (i - 1))
        predict_val.append(x)
    print('预测值是： {0}'.format(predict_val))

    simulate_val_all = simulate_val + predict_val

    #误差计算
    #残差
    residual = data - simulate_val
    relative_error = []
    #相对误差
    for i in range(len(data)):
        x = abs(residual[i])/data[i]
        relative_error.append(x)
    print('残差是： {}'.format(residual))
    print('相对误差是： {}'.format(relative_error))

    #画图
    plt.plot(simulate_val_all,label='simulate value')
    plt.plot(data,label='real value')
    plt.vlines(time[-1],0,max(data),label='predict value')
    plt.xlabel('time')
    plt.ylabel('cases')
    plt.legend()
    plt.show()



#Roll_GM(1,1)
def roll_gm(data,roll_step=4,predict_step=1):
    #数据预处理
    data = data.reshape(1,len(data))
    data1 = [x for y in data for x in y]#用于画图
    data = [x for y in data for x in y]#降维

    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.window(roll_step, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(roll_step))

    count = 0#增加一个计数器，用于记录迭代次数
    a = []
    b = []
    predict_value = []
    simulate_value = []
    for item in data:
        item = opeators.bar_opeators(item)###########这里增加了一个算子处理，看看怎么样子误差更小
        item = np.array(item)#把列表转化为np.array
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
        a.append(coff[0][0])
        b.append(coff[1][0])

        for i in range(len(data_cum) + 1, len(data_cum) + predict_step+1):
            x = (1 - np.exp(a[count])) * (item[0] - b[count] / a[count]) * np.exp(-a[count] * (i - 1))#用count获取a，b
            predict_value.append(x)
        count += 1
    print(predict_value)

    # 误差计算
    # 残差
    residual = []
    data_cal = data1[roll_step:]
    for i in range(len(data_cal)):
        x = data_cal[i] - predict_value[i]
        residual.append(x)
    relative_error = []
    # 相对误差
    for i in range(len(data_cal)):
        x = abs(residual[i]) / data_cal[i]
        relative_error.append(x)
    print('残差是： {}'.format(residual))
    print('相对误差是： {}'.format(relative_error))


    plt.plot(predict_value,label='predict value')
    plt.plot(data1[roll_step:],label='real value')
    plt.xlabel('time')
    plt.ylabel('cases')
    plt.legend()
    plt.show()



#GM(1,3)
def multi_gm(data_1,data_2,data_3,predict_step=1):
    data_1_sim, data_2_sim, data_3_sim = data_1[:-predict_step], data_2[:-predict_step], data_3[:-predict_step]
    data_1_pre, data_2_pre, data_3_pre = data_1[-predict_step:], data_2[-predict_step:], data_3[-predict_step:]
    data_1_cum = np.cumsum(data_1_sim)
    data_2_cum = np.cumsum(data_2_sim)
    data_3_cum = np.cumsum(data_3_sim)
    Y = data_1_sim.reshape(len(data_1_sim), 1)[1:]
    data_2_cum_shaped = data_2_cum.reshape(len(data_2_cum), 1)[1:]
    data_3_cum_shaped = data_3_cum.reshape(len(data_3_cum), 1)[1:]
    data_1_cum_z = []
    for i in range(1, len(data_1_cum)):
        z = 0.5 * (data_1_cum[i] + data_1_cum[i - 1])
        data_1_cum_z.append(z)  # 使用均值形式
    data_1_cum_z = np.array(data_1_cum_z)
    data_1_cum_z1 = -data_1_cum_z.reshape(len(data_1_cum_z), 1)
    B = np.hstack((data_1_cum_z1, data_2_cum_shaped, data_3_cum_shaped))

    coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
    a, b1, b2= coff[0], coff[1], coff[2]
    print(coff)

    simulate_values = []
    simulate_values.append(data_1[0])
    for i in range(1, len(data_1_sim)):
        x = -a*data_1_cum_z[i-1] + b1*data_2_cum[i] + b2*data_3_cum[i]
        simulate_values.append(x)
    print(simulate_values)

    data_2_cum_all = np.cumsum(data_2)
    data_3_cum_all = np.cumsum(data_3)

    predict_values = [213991.27,238567.58]
    # for i in range(len(data_1_cum), len(data_1_cum) + predict_step):
    #     x = -a*data_1_cum_z[i-1] + b1*data_2_cum_all[i] + b2*data_3_cum_all[i]
    #     predict_values.append(x)
    #     data_1_cum = np.append(data_1_cum, data_1_cum[i - 1] + x)
    #     data_1_cum_z = np.append(data_1_cum_z,data_1_cum[i - 1] + x/2)
    print('预测值是： {0}'.format(predict_values))

    simulate_val_all = simulate_values + predict_values
    print(simulate_val_all)



    plt.plot(data_1,label='Real Value')
    plt.plot(simulate_val_all,label='Simulate Value')
    plt.vlines(len(data_1) - predict_step - 1, min(data_1) - 1000, max(data_1), label='predict')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    rel_error = []
    for i in range(1, len(data_1_sim)):
        x = abs(simulate_values[i] - data_1[i]) / data_1[i]
        rel_error.append(x)
    print(rel_error)



#GM（1，N|sin）
# def multi_gm_sin_power(data_1,data_2,data_3,gama1=1.165978,gama2= 0.885304,p=1,predict_step=1):
#     data_1_cum = np.cumsum(data_1)
#     data_2_cum = np.cumsum(data_2)
#     data_3_cum = np.cumsum(data_3)
#     Y = data_1.reshape(len(data_1), 1)[1:]
#     data_2_cum_exp = []
#     for item in data_2_cum:
#         x = item ** gama1
#         data_2_cum_exp.append(x)
#     data_2_cum_exp = np.array(data_2_cum_exp)
#     data_3_cum_exp = []
#     for item in data_3_cum:
#         x = item ** gama2
#         data_3_cum_exp.append(x)
#     data_3_cum_exp = np.array(data_3_cum_exp)
#     data_2_cum_shaped = data_2_cum_exp.reshape(len(data_2_cum), 1)[1:]
#     data_3_cum_shaped = data_3_cum_exp.reshape(len(data_3_cum), 1)[1:]
#     data_1_cum_z = []
#     for i in range(1, len(data_1_cum)):
#         z = 0.5 * (data_1_cum[i] + data_1_cum[i - 1])
#         data_1_cum_z.append(z)  # 使用均值形式
#     data_1_cum_z = np.array(data_1_cum_z)
#     data_1_cum_z = -data_1_cum_z.reshape(len(data_1_cum_z), 1)
#     sin_list = []
#     for i in range(2,len(data_1)+1):
#         x = math.sin(i*p)
#         sin_list.append(x)
#     sin_list = np.array(sin_list)
#     sin_list = sin_list.reshape(len(sin_list),1)
#     ones_list = np.ones(len(data_1)-1).reshape(len(data_1)-1,1)
#     B = np.hstack((data_1_cum_z, data_2_cum_shaped, data_3_cum_shaped,sin_list,ones_list))
#
#     coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
#     a,b1,b2,b3,b4 = coff[0],coff[1],coff[2],coff[3],coff[4]
#     # a,b1,b2 = coff[0],coff[1],coff[2]
#     print(coff)
#
#     simulate_values = []
#     simulate_values.append(data_1[0])
#     for i in range(1,len(data_2_cum)):
#         y = data_2_cum[i]**gama1*b1/(1+0.5*a) + data_3_cum[i]**gama2*b2/(1+0.5*a)
#         x = y - data_1_cum[i-1]*(a/(1+0.5*a))+(b3*math.sin(p*i))/(1+0.5*a)+b4/(1+0.5*a)
#         simulate_values.append(x)
#     print(simulate_values)
#
#
#     # predict_values = []
#     # for i in range(len(data_1_cum),len(data_1_cum)+1+predict_step):
#     #     y = (data_2_cum[5]+958) ** gama1 * b1 / (1 + 0.5 * a) + (data_3_cum[5]+4882) ** gama2 * b2 / (1 + 0.5 * a)
#     #     x = y-data_1_cum[5]*(a/(1+0.5*a))+(b3*math.sin(p*6))/(1+0.5*a)+b4/(1+0.5*a)
#     # predict_values.append(x)
#     # print('预测值是： {0}'.format(predict_values))
#     #
#     # simulate_val_all = simulate_values + predict_values
#
#     plt.plot(data_1, label='Real Value',marker='o')
#     plt.plot(simulate_values, label='Simulate Value',marker='+')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()
#
#     rel_error = []
#     for i in range(1, len(data_1)):
#         x = abs(simulate_values[i] - data_1[i]) / data_1[i]
#         rel_error.append(x)
#     print(rel_error)





def multi_gm_sin(data_1,data_2,data_3,p=1,predict_step=1):
    data_1_cum = np.cumsum(data_1)
    data_2_cum = np.cumsum(data_2)
    data_3_cum = np.cumsum(data_3)
    Y = data_1.reshape(len(data_1), 1)[1:]
    data_2_cum_shaped = data_2_cum.reshape(len(data_2_cum), 1)[1:]
    data_3_cum_shaped = data_3_cum.reshape(len(data_3_cum), 1)[1:]
    data_1_cum_z = []
    for i in range(1, len(data_1_cum)):
        z = 0.5 * (data_1_cum[i] + data_1_cum[i - 1])
        data_1_cum_z.append(z)  # 使用均值形式
    data_1_cum_z = np.array(data_1_cum_z)
    data_1_cum_z = -data_1_cum_z.reshape(len(data_1_cum_z), 1)
    sin_list = []
    for i in range(2,len(data_1)+1):
        x = math.sin(i*p)
        sin_list.append(x)
    sin_list = np.array(sin_list)
    sin_list = sin_list.reshape(len(sin_list),1)
    ones_list = np.ones(len(data_1)-1).reshape(len(data_1)-1,1)
    B = np.hstack((data_1_cum_z, data_2_cum_shaped, data_3_cum_shaped,sin_list,ones_list))

    coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
    a,b1,b2,b3,b4 = coff[0],coff[1],coff[2],coff[3],coff[4]
    # a,b1,b2 = coff[0],coff[1],coff[2]
    print(coff)

    simulate_values = []
    simulate_values.append(data_1[0])
    for i in range(1,len(data_2_cum)):
        y = data_2_cum[i]*b1/(1+0.5*a) + data_3_cum[i]*b2/(1+0.5*a)
        x = y - data_1_cum[i-1]*(a/(1+0.5*a))+(b3*math.sin(p*i))/(1+0.5*a)+b4/(1+0.5*a)
        simulate_values.append(x)
    print(simulate_values)


    predict_values = []
    y = (data_2_cum[5]+958)* b1 / (1 + 0.5 * a) + (data_3_cum[5]+4882)* b2 / (1 + 0.5 * a)
    x = y-data_1_cum[5]*(a/(1+0.5*a))+(b3*math.sin(p*6))/(1+0.5*a)+b4/(1+0.5*a)
    predict_values.append(x)
    print('预测值是： {0}'.format(predict_values))

    simulate_val_all = simulate_values + predict_values

    plt.plot(data_1, label='Real Value',marker='o')
    plt.plot(simulate_val_all, label='Simulate Value',marker='+')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    rel_error = []
    for i in range(1,len(data_1)):
        x = abs(simulate_values[i]-data_1[i])/data_1[i]
        rel_error.append(x)
    print(rel_error)


def multi_gm_power(data_1,data_2,data_3,gama1=1.165978,gama2= 0.885304,predict_step=1):
    data_1_sim, data_2_sim, data_3_sim = data_1[:-predict_step], data_2[:-predict_step], data_3[:-predict_step]
    data_1_pre, data_2_pre, data_3_pre = data_1[-predict_step:], data_2[-predict_step:], data_3[-predict_step:]
    data_1_cum = np.cumsum(data_1_sim)
    data_2_cum = np.cumsum(data_2_sim)
    data_3_cum = np.cumsum(data_3_sim)
    Y = data_1_sim.reshape(len(data_1_sim), 1)[1:]
    data_2_cum_exp = []
    for item in data_2_cum:
        x = item ** gama1
        data_2_cum_exp.append(x)
    data_2_cum_exp = np.array(data_2_cum_exp)
    data_3_cum_exp = []
    for item in data_3_cum:
        x = item ** gama2
        data_3_cum_exp.append(x)
    data_3_cum_exp = np.array(data_3_cum_exp)
    data_2_cum_shaped = data_2_cum_exp.reshape(len(data_2_cum), 1)[1:]
    data_3_cum_shaped = data_3_cum_exp.reshape(len(data_3_cum), 1)[1:]
    data_1_cum_z = []
    for i in range(1, len(data_1_cum)):
        z = 0.5 * (data_1_cum[i] + data_1_cum[i - 1])
        data_1_cum_z.append(z)  # 使用均值形式
    data_1_cum_z = np.array(data_1_cum_z)
    data_1_cum_z = -data_1_cum_z.reshape(len(data_1_cum_z), 1)
    B = np.hstack((data_1_cum_z, data_2_cum_shaped, data_3_cum_shaped))

    coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
    a,b1,b2 = coff[0],coff[1],coff[2]
    print(coff)

    simulate_values = []
    simulate_values.append(data_1[0])
    for i in range(1,len(data_2_cum)):
        y = data_2_cum[i]**gama1*b1/(1+0.5*a) + data_3_cum[i]**gama2*b2/(1+0.5*a)
        x = y - data_1_cum[i-1]*(a/(1+0.5*a))
        simulate_values.append(x)
    print(simulate_values)

    data_2_cum_all = np.cumsum(data_2)
    data_3_cum_all = np.cumsum(data_3)

    predict_values = []
    for i in range(len(data_1_cum), len(data_1_cum) + predict_step):
        y = (data_2_cum_all[i]) ** gama1 * b1 / (1 + 0.5 * a) + (data_3_cum_all[i]) ** gama2 * b2 / (1 + 0.5 * a)
        x = y - data_1_cum[i - 1] * (a / (1 + 0.5 * a))
        predict_values.append(x)
        data_1_cum = np.append(data_1_cum, data_1_cum[i - 1] + x)
    print('预测值是： {0}'.format(predict_values))


    simulate_val_all = simulate_values + predict_values
    print(simulate_val_all)

    plt.plot(data_1, label='Real Value',marker='o')
    plt.plot(simulate_val_all, label='Simulate Value',marker='+')
    plt.vlines(len(data_1) - predict_step - 1, min(data_1) - 1000, max(data_1), label='predict')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    rel_error = []
    for i in range(1, len(data_1_sim)):
        x = abs(simulate_values[i] - data_1[i]) / data_1[i]
        rel_error.append(x)
    print(rel_error)


def multi_gm_sin_power(data_1,data_2,data_3,gama1=1.7254,gama2=0.76642,p=1.5,predict_step=1):
    data_1_sim,data_2_sim,data_3_sim = data_1[:-predict_step],data_2[:-predict_step],data_3[:-predict_step]
    data_1_pre,data_2_pre,data_3_pre = data_1[-predict_step:],data_2[-predict_step:],data_3[-predict_step:]
    data_1_cum = np.cumsum(data_1_sim)
    data_2_cum = np.cumsum(data_2_sim)
    data_3_cum = np.cumsum(data_3_sim)
    Y = data_1_sim.reshape(len(data_1_sim), 1)[1:]
    data_2_cum_exp = []
    for item in data_2_cum:
        x = item ** gama1
        data_2_cum_exp.append(x)
    data_2_cum_exp = np.array(data_2_cum_exp)
    data_3_cum_exp = []
    for item in data_3_cum:
        x = item ** gama2
        data_3_cum_exp.append(x)
    data_3_cum_exp = np.array(data_3_cum_exp)
    data_2_cum_shaped = data_2_cum_exp.reshape(len(data_2_cum), 1)[1:]
    data_3_cum_shaped = data_3_cum_exp.reshape(len(data_3_cum), 1)[1:]
    data_1_cum_z = []
    for i in range(1, len(data_1_cum)):
        z = 0.5 * (data_1_cum[i] + data_1_cum[i - 1])
        data_1_cum_z.append(z)  # 使用均值形式
    data_1_cum_z = np.array(data_1_cum_z)
    data_1_cum_z = -data_1_cum_z.reshape(len(data_1_cum_z), 1)
    sin_list = []
    for i in range(2,len(data_1_sim)+1):
        x = math.sin(i*p)
        sin_list.append(x)
    sin_list = np.array(sin_list)
    sin_list = sin_list.reshape(len(sin_list),1)
    ones_list = np.ones(len(data_1_sim)-1).reshape(len(data_1_sim)-1,1)
    B = np.hstack((data_1_cum_z, data_2_cum_shaped, data_3_cum_shaped,sin_list,ones_list))

    coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
    a,b1,b2,b3,b4 = coff[0],coff[1],coff[2],coff[3],coff[4]
    # a,b1,b2 = coff[0],coff[1],coff[2]
    print(coff)

    simulate_values = []
    simulate_values.append(data_1_sim[0])
    for i in range(1,len(data_2_cum)):
        y = data_2_cum[i]**gama1*b1/(1+0.5*a) + data_3_cum[i]**gama2*b2/(1+0.5*a)
        x = y - data_1_cum[i-1]*(a/(1+0.5*a))+(b3*math.sin(p*i))/(1+0.5*a)+b4/(1+0.5*a)
        # u = b1/a
        # o = b2/a
        # y = (data_2_cum[i]**gama1)*u + (data_3_cum[i]**gama2)*o
        # m = b3 / (a ** 2 + p ** 2)
        # x = (data_1[0]-m*(a*np.sin(p)-p*np.cos(p))-(b4/a)-y)*np.exp(-a*(i-1))+m*(a*np.sin(p*i)-p*np.cos(p*i))+b4/a
        simulate_values.append(x)
    print(simulate_values)

    # lt = []
    # lt.append(data_1[0])
    # for i in range(1,len(simulate_values)):
    #     x = simulate_values[i] - simulate_values[i-1]
    #     lt.append(x)
    # simulate_values = lt


    data_2_cum_all = np.cumsum(data_2)
    data_3_cum_all = np.cumsum(data_3)

    predict_values = []
    for i in range(len(data_1_cum),len(data_1_cum)+predict_step):
        y = (data_2_cum_all[i])**gama1*b1/(1+0.5*a) + (data_3_cum_all[i])**gama2*b2/(1+0.5*a)
        x = y - data_1_cum[i-1]*(a/(1+0.5*a))+(b3*math.sin(p*i))/(1+0.5*a)+b4/(1+0.5*a)
        predict_values.append(x)
        data_1_cum = np.append(data_1_cum,data_1_cum[i-1]+x)
    print('预测值是： {0}'.format(predict_values))

    simulate_val_all = simulate_values + predict_values
    print(list(simulate_val_all))

    plt.plot(data_1, label='Real Value',marker='o')
    plt.plot(simulate_val_all, label='Simulate Value',marker='+')
    plt.vlines(len(data_1)-predict_step-1, min(data_1)-1000, max(data_1), label='predict')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    rel_error = []
    for i in range(1, len(data_1_sim)):
        x = abs(simulate_values[i] - data_1[i]) / data_1[i]
        rel_error.append(x)
    print(rel_error)



# multi_gm_sin_power(data,data1,data2,gama1=0.7,gama2=1,p=0.3)






