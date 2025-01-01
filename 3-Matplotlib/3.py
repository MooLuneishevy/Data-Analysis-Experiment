import numpy as np  # 导入numpy库
import matplotlib.pyplot as plt  # 导入matplotlib库
np.random.seed(0)  # 设置随机种子为0
mu = 200  # 设定正态分布的均值
sigma = 25  # 设定正态分布的标准差
x = np.random.normal(mu, sigma, size=100)  # 从正态分布中产生100个随机样本
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))  # 创建图中包含两个子图 ncols=2表示并排放置两列子图 设置图形大小为(8,4)
ax0.hist(x, 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75)  # 在第一个子图ax0中绘制直方图 20为柱形数 normed=1即直方图归一化 facecolor为填充颜色 alpha为透明度
ax0.set_title('stepfilled')  # 设置第一个子图的标题为stepfilled
bins = [100, 150, 180, 195, 205, 220, 250, 300]  # 定义自定义的bin边界
ax1.hist(x, bins, density=True, histtype='bar', rwidth=0.8)  # 在第二个子图ax1中绘制直方图 使用自定义的bins normed=1即直方图归一化 rwidth设置柱形宽度
ax1.set_title('unequal bins')  # 设置第二个子图的标题为unequal bins
fig.tight_layout()  # 调整子图参数 使之填充整个图像区域
plt.show()  # 显示图表