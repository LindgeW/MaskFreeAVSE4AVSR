import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class VideoModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(VideoModel, self).__init__()
        self.feature_extractor = nn.Linear(1024, feature_dim)  # 示例特征提取层
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

# 相似度计算
def cosine_similarity(x, y):
    return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)

# 损失函数示例
def contrastive_loss(features, labels, label_embeddings, margin=1.0):
    similarities = cosine_similarity(features, label_embeddings)
    print(features.shape, label_embeddings.shape, similarities.shape)
    positive_sim = torch.gather(similarities, 1, labels.view(-1, 1))
    negative_sim = similarities.max(dim=1)[0]
    loss = F.relu(margin - positive_sim + negative_sim).mean()
    return loss

# 示例输入和标签
video_input = torch.randn(8, 1024)  # 假设输入视频表征
labels = torch.randint(0, 5, (8,))  # 5个类别

# 初始化模型
model = VideoModel(feature_dim=128, num_classes=5)
features = model(video_input)
loss = contrastive_loss(features, labels, model.label_embeddings)

print(f'Loss: {loss.item()}')