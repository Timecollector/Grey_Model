2022.3.30更新
- 将灰色预测模型移动至GreyForecasting文件夹中
- 新增了GreyIncidence文件夹，已更新部分灰色关联度模型，后续将继续更新



### **对部分代码进行了重写**

### **如需使用GM(1,1)模型、GM(1,N)模型、GM(1,N|sin)幂模型，请使用文件`GreyForecasting`中的`gm11.py`、`gm1n.py`、`pgm1nsin.py`代码**



## 灰色预测模型

- **灰色多变量周期幂模型**

  - 针对`Grey_Model.py`中最小二乘部分的一些错误进行了修改

  - 针对`Grey_Model.py`中`pgm1nsin`模型代码进行了重写与优化

  - 使`pgm1nsin`模型可以自行选择哪一变量进行幂指数运算或周期正弦运算


具体使用方法如下：

```
定义GM(1,N|sin)幂模型

参数设置:
sys_data:系统行为序列
rel_p_data:相关因素序列（指数因素）
rel_s_data:相关因素序列（周期因素）
predict_step:预测步长，会截取上面三组数据的最后predict_step行数据作为预测
gamma:幂指数
p:周期系数
使用方法：
1.实例化对象 model = pgm1nsin(sys_data=sys_data,rel_p_data=rel_p_data,rel_s_data=rel_s_data)\n
2.训练模型  model.fit()\n
3.进行预测  model.predict
```

- **灰色单变量预测模型**
  - 对代码进行了重写与优化


```python
model = gm11(data,predstep=2)
fitted_values = model.fit()
predict_values = model.predict()
```

- **灰色多变量预测模型**
  - 更新了GM(1,N)的拟合部分，预测部分有空写完上传


- **灰色多变量幂模型**
  - 参考*王正新.灰色多变量GM(1,N)幂模型及其应用[J].系统工程理论与实践,2014,34(09):2357-2363.*编写的代码
  - **后续会按照灰色多变量周期幂模型的格式重构代码**，当前如需使用，请使用`Grey_Model_v2.py`中的**pgm1ns模型**，令其**p=1**

```python
model = pgm1ns()
model.fit(data, predict_step=2, gama=[1.7254, 0.86642], p=0)
values = model.get_all_sim_val()
```

- **测试数据**

在`Power.xlsx`文件中保存着江苏省季度用电量、温度、GDP等数据，用于对模型进行测试

| 用电量  | GDP     | 温度  |
| ------- | ------- | ----- |
| 1170.66 | 1727.29 | 7.80  |
| 1238.95 | 2380.31 | 21.49 |
| 1331.82 | 2112.61 | 25.55 |
| 1271.11 | 2600.54 | 11.96 |



## 灰色关联模型

当前更新了三个关联模型，分别是：

- 邓聚龙教授的绝对关联模型
- 叶莉莉等在[*叶莉莉,谢乃明,罗党.累积时滞非线性ATNDGM(1,N)模型构建及应用[J].系统工程理论与实践,2021,41(09):2414-2427.*]中构建的关联模型
- 王俊杰在[*王俊杰. 时滞性与周期性的灰建模技术研究及其在雾霾治理中的应用[D].南京航空航天大学,2018.DOI:10.27239/d.cnki.gnhhu.2018.000204.*]中构建的等周期关联模型



- **灰色绝对关联模型**

代码在`Greyinc.py`中

```python
data_in = mean_process(data_in)
data_out = mean_process(data_out)
inc_mat = gery_inci(data_in, data_in)
```

- **叶莉莉等的灰色关联模型（主要用于寻找时滞期）**

代码在`Time_lag_model.py`中

```python
# 支持相关因素与系统因素都有多个，返回一个矩阵，行是相关因素，列是系统因素
data_input = data.iloc[14:,1:9]
data_output = data.iloc[14:,9:]
inc_mat = time_lag_inc(data_input,data_output,3)
```

- **等周期关联模型**

代码在`GreyDelayTrendModeltl.py`中

```
 # 仅支持一个相关因素与一个系统因素，返回不同趋势t下的点关联度，行为不同趋势，矩阵最后一列是该趋势下的平均关联度
 data1 = mean_process(data.iloc[:, 0:1])
 data2 = mean_process(data.iloc[:, 1:2])
 a = gdtep_t(data1, data2)
```

