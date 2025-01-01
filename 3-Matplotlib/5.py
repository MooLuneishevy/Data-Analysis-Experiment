import numpy as np  # 导入numpy库
import matplotlib.pyplot as plt  # 导入matplotlib库
fig = plt.figure(0)  # 创建图形对象1 使用编号0
x = np.arange(10.0)  # 创建x轴数据 从0到9
y = np.sin(np.arange(10.0) / 20.0 * np.pi)  # 计算y轴数据 生成一组正弦随机值
plt.errorbar(x, y, yerr=0.1)  # 绘制数据1的误差条 yerr设置误差范围为0.1
y = np.sin(np.arange(10.0) / 20.0 * np.pi) + 1  # 将y+1来得到新的随机数据
plt.errorbar(x, y, yerr=0.1, uplims=True)  # 绘制数据2的误差条 uplims=True表示y的上限有效
y = np.sin(np.arange(10.0) / 20.0 * np.pi) + 2  # 将y+2来得到新的随机数据
upperlimits = np.array([1, 0] * 5)  # 上限设定为1 0交替
lowerlimits = np.array([0, 1] * 5)  # 下限设定为0 1交替
plt.errorbar(x, y, yerr=0.1, uplims=upperlimits, lolims=lowerlimits)  # 绘制数据3的误差条 上限和下限有效
plt.xlim(-1, 10)  # 设置x轴的范围 从-1到10
fig = plt.figure(1)  # 创建图形对象2 使用编号 1
x = np.arange(10.0) / 10.0  # 创建x轴数据 从0到0.9
y = (x + 0.1) ** 2  # 计算y轴数据 y=(x+0.1)^2
plt.errorbar(x, y, xerr=0.1, xlolims=True)  # 绘制数据1的误差条 xerr设置误差 xlolims=True表示下限有效
y = (x + 0.1) ** 3  # 计算y轴数据 y=(x+0.1)^3
plt.errorbar(x + 0.6, y, xerr=0.1, xuplims=upperlimits, xlolims=lowerlimits)  # 绘制数据2的误差条 上限和下限有效
y = (x + 0.1) ** 4  # 计算y轴数据 y=(x+0.1)^4
plt.errorbar(x + 1.2, y, xerr=0.1, xuplims=True)  # 绘制数据3的误差条 uplims=True表示上限有效
plt.xlim(-0.2, 2.4)  # 设置x轴的范围 从-0.2到2.4
plt.ylim(-0.1, 1.3)  # 设置y轴的范围 从-0.1到1.3
plt.show()  # 显示绘制图形