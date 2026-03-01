import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# 加载数据
distance_df = pd.read_csv("dataset\pems04\distance.csv")
data = np.load("dataset\pems04\pems04.npz")
traffic_data = data['data']

# 构建邻接矩阵
num_nodes = traffic_data.shape[1]
adj_matrix = np.zeros((num_nodes, num_nodes))
for _, row in distance_df.iterrows():
    from_node = int(row['from'])
    to_node = int(row['to'])
    cost = float(row['cost'])
    adj_matrix[from_node, to_node] = cost

# 可视化邻接矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(adj_matrix, cmap="viridis")
# plt.title("Adjacency Matrix Heatmap (PEMS04)")
# plt.xlabel("To Node")
# plt.ylabel("From Node")
# plt.tight_layout()
# plt.show()

# 构造图
G = nx.from_numpy_array((adj_matrix > 0).astype(int), create_using=nx.DiGraph)
degrees = list(dict(G.degree()).values())

# 可视化度分布
plt.figure(figsize=(6, 4))
sns.histplot(
    degrees,
    bins=range(min(degrees), max(degrees) + 2),  # 整数分箱，确保中心对齐
    kde=True,
    color='orange',
    edgecolor='black',
    discrete=True,
    shrink=1  # 不压缩条宽
)
# plt.title("Node degree distribution of PeMS04")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.xticks(range(min(degrees), max(degrees) + 1))
plt.tight_layout()
plt.show()

# 可视化单节点时序
sample_node = 0
time_series = traffic_data[:, sample_node, 0]

plt.figure(figsize=(8, 3))
plt.plot(time_series[:500], color='orange')
# plt.title("Total flow diagram of PeMS04")
plt.xlabel("Time Step")
plt.ylabel("Total Flow")
plt.tight_layout()
plt.show()
