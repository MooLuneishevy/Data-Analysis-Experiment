'''
任务1.3
根据任务1.2的结果 统计各省级行政单位每天新冠病人的住院人数 将结果保存为 task1_3.csv
第一行为字段名 按省份 日期 住院人数的次序分别放在 A 列~C 列
在实验报告中给出实现方法的相关描述 并列表给出湖北 广东 上海每月 20 日的统计结果
'''
import openpyxl
import pandas as pd
# 数据读入
result_2 = pd.read_csv('task1_2.csv')
# 计算住院人数
result_2['住院人数'] = result_2['累计确诊'] - result_2['累计治愈'] - result_2['累计死亡']
# 选择输出列并保存结果
result = result_2[['省份', '日期', '住院人数']]
result.to_csv('task1_3.csv', index=False)
# 提取湖北 广东 上海每月20日的数据
result_search = result[result['省份'].isin(['湖北', '广东', '上海']) & (pd.to_datetime(result['日期']).dt.day == 20)]
print(result_search)