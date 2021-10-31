# encoding:utf-8
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# # 获取Karate俱乐部数据的邻接表，以csv文件存储
columns = ['Source', 'Target']
data = pd.read_csv('data.csv', names=columns, header=None,delimiter='	')

# 调用networkx画图
graph = nx.karate_club_graph()
data_len = len(data)

# 给graph中添加边，即用户关系
for i in range(data_len):
    graph.add_edge(data.iloc[i]['Source'], data.iloc[i]['Target'])

# 输出每个节点的度
print(graph.degree())

# 调用nx.draw()方法绘制图片
nx.draw(graph, with_labels=True)

# 保存图片
plt.savefig('karate_club.png')

# 展示图片
plt.show()
