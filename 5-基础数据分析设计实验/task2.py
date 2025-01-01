'''
任务1.2
根据任务 1.1 的结果 并结合附件 1 城市省份对照表 统计各省级行政单位按日新增和累计数据 将结果保存为 task1_2.csv
第一行为字段名 按省份 日期 新增确诊人数 新增治愈人数 新增死亡人数 累计确诊人数 累计治愈人数 累计死亡人数的次序分别放在 A 列~H 列
在实验报告中给出实现方法的相关描述 并列表给出湖北 广东 河北每月 15 日的统计结果
'''
import openpyxl
import pandas as pd
# 数据读取
result_1 = pd.read_csv('task1_1.csv')
data_province = pd.read_excel('新冠疫情分析数据-附件1.xlsx', sheet_name='城市省份对照表')
data = pd.merge(result_1, data_province, on='城市')
# 计算累计数据
result = data.groupby(['省份', '日期']).agg(累计确诊=('累计确诊', 'sum'),累计治愈=('累计治愈', 'sum'),累计死亡=('累计死亡', 'sum')).reset_index()
# 计算新增数据
result.sort_values(by=['省份', '日期'], inplace=True)
result['新增确诊'] = result.groupby('省份')['累计确诊'].diff().fillna(0)
result['新增治愈'] = result.groupby('省份')['累计治愈'].diff().fillna(0)
result['新增死亡'] = result.groupby('省份')['累计死亡'].diff().fillna(0)
# 选择输出列 保存结果
result = result[['省份', '日期', '新增确诊', '新增治愈', '新增死亡', '累计确诊', '累计治愈', '累计死亡']]
result.to_csv('task1_2.csv', index=False)
# 提取湖北 广东 河北每月15日的数据
result_search = result[result['省份'].isin(['湖北', '广东', '河北']) & (pd.to_datetime(result['日期']).dt.day == 15)]
print(result_search)