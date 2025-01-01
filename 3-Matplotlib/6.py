import matplotlib.pyplot as plt  # 导入matplotlib库
import numpy as np  # 导入numpy库
font1 = {'weight': 'normal','size': 19,}  # 定义字体样式字典
font2 = {'weight': 'normal','size': 12,}  # 定义字体样式字典
def auto_label(rects):  # 在条形图上添加数值标签
    for rect in rects:  # 遍历每个矩形条
        height = rect.get_height()  # 获取矩形的高度
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')   # 在矩形顶部其值 获取矩形中心的坐标 竖直偏移3个点 设定偏移 坐标文本水平居中与底部对齐
def auto_text(rects):  # 在条形图的左侧显示数值
    for rect in rects:  # 遍历每个矩形条
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom', rotation=45)  # 在矩形左侧显示高度 设置旋转45度
labels = ['UBCF', 'IBCF', 'MF-eALS', 'MF-BPR', 'NCF']  # 模型标签
or_means = [0.2503, 0.2630, 0.2714, 0.2689, 0.2834]  # 原始模型均值
md_means = [0.3142, 0.3255, 0.3201, 0.3307, 0.3401]  # 使用MMD后的模型均值
# or_means = [0.2432, 0.2354, 0.2522, 0.2303, 0.2511]
# md_means = [0.2899, 0.2906, 0.3001, 0.2716, 0.3012]
# or_means = [0.3212, 0.3121, 0.3142, 0.3065, 0.3220]
# md_means = [0.3660, 0.3642, 0.3714, 0.3356, 0.3746]
# or_means = [0.3206, 0.3116, 0.3210, 0.2946, 0.3301]
# md_means = [0.3713, 0.3532, 0.3571, 0.3611, 0.3812]
index = np.arange(len(labels))  # 定义x轴索引
width = 0.3  # 设定条形的宽度
fig, ax = plt.subplots()  # 创建图形和坐标轴对象
rect1 = ax.bar(index - width / 2, or_means, edgecolor="k", color='red', width=width, label='original')  # 绘制原始模型的条形图
rect2 = ax.bar(index + width / 2, md_means, edgecolor="k", color='blue', width=width, label='with MMD')  # 绘制使用MMD后的模型的条形图
# ax.set_title('Scores by gender')
ax.set_xticks(ticks=index)  # 设置 x 轴刻度
ax.set_xticklabels(labels)  # 设置 x 轴标签
ax.set_ylabel('HR@15', font1)  # 设置 y 轴标签
ax.set_xlabel('models', font1)  # 设置 x 轴标签
ax.set_ylim(0.22, 0.38)  # 设置 y 轴范围
# auto_label(rect1)
# auto_label(rect2)
auto_text(rect1)  
auto_text(rect2)  
# ax.legend(loc='upper right', frameon=False)
ax.legend(bbox_to_anchor=[0.7, 1], prop=font2)  # 设置图例位置 使用自定义字体样式
# fig.tight_layout()
plt.savefig("test3.png", dpi=300)  # 保存图形为PNG文件 分辨率设置为300dpi
plt.show()  # 显示绘制的图形