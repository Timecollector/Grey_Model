# Grey_Model
现在做了GM(1,1),滚动的GM(1,1)，由GM(1,N)衍生的GM(1,N)幂模型、GM(1,N|sin)幂模型以及GM(1,N|sin)模型

使用方法：
import Grey_Model as gm

gmsin = gm.pgmns()#实例化
gmsin.fit(data,predict_step=2,gama=[1.2,0.9],p=2)#拟合函数
接下来就可以调用类中的方法了，具体方法可以看代码

