# -*- coding: utf-8 -*-
# 绘制人物关系频数统计条形图
import pandas as pd
import matplotlib.pyplot as plt

# 读取EXCEL数据
df = pd.read_excel('data_example.xls')
label_list = list(df['关系'].value_counts().index)
num_list = df['关系'].value_counts().tolist()

# Mac系统设置中文字体支持
# plt.rcParams["font.family"] = 'Arial Unicode MS'
# Windows系统设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 利用Matplotlib模块绘制条形图
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
# plt.ylim(0, 800) # y轴范围
plt.ylabel("数量")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xticks(rotation=45)     # x轴的标签旋转45度
plt.xlabel("实体关系")
plt.title("实体关系频数统计")
plt.legend()

# 条形图的文字说明
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

# plt.show()
plt.savefig('./bar_chart.png')


plt.figure(figsize = (10, 4))

plt.subplot(1, 3, 1)
plt.title("Training Set")
sns.histplot(train,x='length',hue='label',kde=True)

plt.subplot(1, 3, 3)
plt.title("Testing Set")
sns.histplot(test,x='length',hue='label',kde=True)
plt.show()