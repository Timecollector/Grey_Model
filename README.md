# **对部分代码进行了重写**

# **如需使用GM(1,1)模型、GM(1,N)模型、GM(1,N|sin)幂模型，请使用`gm11.py`、`gm1n.py`、`pgm1nsin.py`中的代码**





# 灰色多变量周期幂模型

# A novel multivariate grey model for forecasting periodic oscillation time series

- 针对`Grey_Model.py`中最小二乘部分的一些错误进行了修改
- 针对`Grey_Model.py`中`pgm1nsin`模型代码进行了重写与优化
- 使`pgm1nsin`模型可以自行选择哪一变量进行幂指数运算或周期正弦运算

具体使用方法如下：

```
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
```



# 灰色单变量预测模型

- 对代码进行了重写与优化

# 灰色多变量预测模型

- 更新了GM(1,N)的拟合部分，预测部分有空写完上传

# 测试数据

在`Power.xlsx`文件中保存着江苏省季度用电量、温度、GDP等数据，用于对模型进行测试

| 用电量  | GDP     | 温度  |
| ------- | ------- | ----- |
| 1170.66 | 1727.29 | 7.80  |
| 1238.95 | 2380.31 | 21.49 |
| 1331.82 | 2112.61 | 25.55 |
| 1271.11 | 2600.54 | 11.96 |

