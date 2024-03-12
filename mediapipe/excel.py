import pandas as pd

# 读取源表格数据
source_file = r'C:\Users\lenovo\Desktop\学生成绩大表\2021\东北林业大学+班级成绩表+2022-2023-2+化学(中外合作办学)2021-01.xls'
df = pd.read_excel(source_file)

# 目标列名称列表
target_columns = [
    '结构化学与光谱实验',
    '物质性质与分析化学实验',
    '仪器分析及实验',
    '基础化学实验4',
    '化学生物学',
    '有机立体化学',
    '中国近现代史纲要'
]

# 选择目标列
selected_columns = df[target_columns]

# 将选定的列保存到新的表格文件
selected_columns.to_excel('目标列.xlsx', index=False)