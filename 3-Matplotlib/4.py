import numpy as np  # 导入numpy库
import matplotlib.pyplot as plt  # 导入matplotlib库
np.random.seed(0)  # 设置随机种子为0
n_bins = 10  # 定义直方图中的柱形数量为10
x = np.random.randn(1000, 3)  # 生成大小为(1000,3) 服从标准正态分布的数组
fig, axes = plt.subplots(nrows=2, ncols=2)  # 创建一个包含2行2列的子图
ax0, ax1, ax2, ax3 = axes.flatten()  # 将子图轴对象展开
colors = ['red', 'tan', 'lime']  # 定义直方图的颜色列表
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors)  # 在第一个子图ax0中绘制直方图
ax0.legend(prop={'size': 10})  # 添加图例 设定字体大小为10
ax0.set_title('bars with legend')  # 设置第一个子图的标题为bars with legend
ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)  # 在第二个子图ax1中绘制堆叠的直方图
ax1.set_title('stacked bar')  # 设置第二个子图的标题为stacked bar
ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)  # 在第三个子图ax2中绘制不填充的堆叠直方图
ax2.set_title('stack step (unfilled)')  # 设置第三个子图的标题为stack step (unfilled)
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]  # 创建一个包含不同样本大小的列表
ax3.hist(x_multi, n_bins, histtype='bar')  # 在第四个子图ax3中绘制不同样本大小的直方图
ax3.set_title('different sample sizes')  # 设置第四个子图的标题为different sample sizes
fig.tight_layout()  # 调整子图参数 进行布局优化
plt.show()  # 显示最终绘制的图形