import torch
import numpy as np
from sklearn.cluster import KMeans


max_num_queries = 50
feat = torch.randn(800, 100).detach().cpu().numpy()

# kmeans = KMeans(n_clusters=2, random_state=0).fit(feat)
# 获取质心和标签
# centroids = torch.tensor(kmeans.cluster_centers_)
# labels = torch.tensor(kmeans.labels_)

kmeans = KMeans(n_clusters=max_num_queries, random_state=0)
labels = kmeans.fit_predict(feat)  # (n_samples, n_features)
# 统计每个聚类的样本数量
cluster_counts = np.bincount(labels)
# 根据聚类样本数量选择查询向量数量
num_queries = np.sum(cluster_counts > 10)

print(num_queries)