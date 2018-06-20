# 作业：通过神经网络预测房价
by Grant, 2018/6/15

## 文件说明
predict-house-prices.ipynb: 完成的Jupyter notebook文件

predict-house-prices.html: notebook页面转存成的html格式

optim_compare.jpg: 几种不同的optim方法的效能比较

## 几点观察
增加模型的深度对执行结果没有明显的影响

(尝试过4, 5, 6层，效果基本和3层相同)


增加迭代次数e或增加learning rate会达到较好效果，基本上在e * lr = 400时达到最优，接下来会出现误差不降反升的情况


