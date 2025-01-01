'''
任务1.4
假设新冠病人的传播半径为 1 km 根据附件 1 A市涉疫场所 在平面图中分别绘制该市第 6 天和第 10 天的疫情传播风险区域
并在实验报告中给出分析和实现过程
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 数据读入 预处理
data = pd.read_excel('新冠疫情分析数据-附件1.xlsx',sheet_name='A市涉疫场所分布')
data['通报日期'] = data['通报日期'].astype(int)
# 筛选出第6天和第10天的数据
data_day6 = data[data['通报日期'] == 6]
data_day10 = data[data['通报日期'] == 10]
# 创建绘图函数
def plot_risk_area(data, day):
    plt.figure(figsize=(10, 10))
    plt.title(f'A city\'s epidemic transmission risk area - day {day}')
    # 对每个疫情场所绘制传播区域
    for index, row in data.iterrows():
        radius = 1
        circle = plt.Circle((row['横坐标（公里）'], row['纵坐标（公里）']), radius, color='r', alpha=0.2)
        # 添加圆形到绘图中
        plt.gca().add_artist(circle)
        plt.plot(row['横坐标（公里）'], row['纵坐标（公里）'], 'ro')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()
# 绘制第6天的传播风险区域
plot_risk_area(data_day6, 6)
# 绘制第10天的传播风险区域
plot_risk_area(data_day10, 10)