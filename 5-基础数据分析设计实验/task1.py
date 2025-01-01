'''
任务1.1
根据附件 1 城市疫情 中的数据统计各城市自首次通报确诊病例后至 6 月 30 日的每日累计确诊人数 累计治愈人数和累计死亡人数
将结果保存为 task1_1.csv 第一行为字段名 按城市 日期 累计确诊人数 累计治愈人数 累计死亡人数的次序分别放在 A 列~E 列
在实验报告中给出实现方法的相关描述 并列表给出武汉 深圳 保定每月 10 25 日的统计结果
'''
import openpyxl
import pandas as pd
# 数据读入 预处理
data = pd.read_excel('新冠疫情分析数据-附件1.xlsx', sheet_name='城市疫情')
data['日期'] = pd.to_datetime(data['日期'])
# 确定时间范围 生成时间序列
min_date = data['日期'].min()
max_date = data['日期'].max()
all_dates = pd.date_range(start=min_date, end=max_date)
# 按城市分组 补全日期
result = []
for city, group in data.groupby('城市'):
    group = group.set_index('日期').reindex(all_dates).fillna(0).reset_index()
    group['城市'] = city
    group.rename(columns={'index': '日期'}, inplace=True)
    group['累计确诊'] = group['新增确诊'].cumsum()
    group['累计治愈'] = group['新增治愈'].cumsum()
    group['累计死亡'] = group['新增死亡'].cumsum()
    result.append(group)
result = pd.concat(result, ignore_index=True)
# 选择输出列 保存结果
result = result[['城市', '日期', '累计确诊', '累计治愈', '累计死亡']]
result.to_csv('task1_1.csv', index=False)
# 提取武汉 深圳 保定每月10日和25日的数据
result_search = result[result['城市'].isin(['武汉', '深圳', '保定']) & (result['日期'].dt.day.isin([10, 25]))]
print(result_search)