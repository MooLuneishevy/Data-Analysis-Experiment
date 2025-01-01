import matplotlib.pyplot as plt  # 导入matplotlib库
import numpy as np  # 导入numpy库
plt.rcdefaults()  # 重置matplotlib库的默认设置
fig, ax = plt.subplots()  # 创建图形与坐标轴对象
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')  # 定义人名元组
y_pos = np.arange(len(people))  # 创建一个数组 表示y轴的位置 从0到4
performance = 3 + 10 * np.random.rand(len(people))  # 随机生成每个人的Performance 在3到13之间
error = np.random.rand(len(people))  # 生成每个人的误差
ax.barh(y_pos, performance, xerr=error, align='center', color='green', ecolor='black')  # 绘制水平条形图 xerr用于表示误差 align进行对齐 color为条形颜色 ecolor为误差颜色
ax.set_yticks(y_pos)  # 设置y轴的刻度
ax.set_yticklabels(people)  # 设置y轴的标签
ax.invert_yaxis()  # 反转y轴
ax.set_xlabel('Performance')  # 设置x轴的标签为Performance
ax.set_title('How fast do you want to go today?')  # 设置图表的标题
plt.show()  # 显示图表