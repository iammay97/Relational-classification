# -*- coding: utf-8 -*-
# Plotting Relational Frequency Statistics Bar Charts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read EXCEL data
df = pd.read_excel('data_example.xls')
label_list = list(df['relationship'].value_counts().index)
num_list = df['relationship'].value_counts().tolist()

# Mac system set Chinese font support
# plt.rcParams["font.family"] = 'Arial Unicode MS'
# Windows system setup Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# Plotting Bar Graphs with the Matplotlib Module
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
# plt.ylim(0, 800) # y-axis range
plt.ylabel("amount")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xticks(rotation=45)     # The labels on the x-axis are rotated 45 degrees
plt.xlabel("Entity Relationships")
plt.title("Frequency statistics for entity relationships")
plt.legend()

# Textual description of the bar chart
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

# plt.show()
plt.savefig('./bar_chart.png')


plt.figure(figsize = (10, 4))

plt.subplot(1, 3, 1)
plt.title("Training Set")
sns.histplot(train, x='length',hue='label',kde=True)

plt.subplot(1, 3, 3)
plt.title("Testing Set")
sns.histplot(test,x='length',hue='label',kde=True)
plt.show()