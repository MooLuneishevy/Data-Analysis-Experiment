import numpy as np  # 导入numpy库
import matplotlib.pyplot as plt  # 导入matplotlib库
x = np.linspace(0, 1, 500)  # 创建一个数组x 包含从0到1区间内 500个均匀分布的点
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)  # 计算函数y的值 y=sin(4*pi*x)*exp(-5*x)
fig, ax = plt.subplots()  # 创建图形和坐标轴对象
ax.fill(x, y, zorder=10)  # 填充y轴与x轴 zorder用于控制绘制的层级
ax.grid(True, zorder=5)  # 开启grid 设定zorder为5 即位于上面填充的下方
plt.show()  # 显示图表