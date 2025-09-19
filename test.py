'''
import librosa
import soundfile as sf
import numpy as np

fn = r'D:\LipData\GRID\audio\s1\bwwbzp.wav'
sr = 16000
y, sr = librosa.load(fn, sr=sr)  # 16kHz
# melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=200, n_mels=80)  #转换到对数刻度
# print(melspec.shape)
# melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=800, hop_length=160, n_mels=80)  #转换到对数刻度
# print(melspec.shape)
noise = np.random.normal(0, 0.05, y.shape)
y = y + noise
y0 = y / np.max(np.abs(y))
sf.write('max_abs_noise.wav', y0, sr)
y1 = (y - np.mean(y)) / np.std(y)
sf.write('mean_std_noise.wav', y1, sr)
print('Done!')
'''


'''
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)
        self.layer4 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        y = torch.relu(self.layer2(x))
        z = self.layer3(y)
        return y, z

# 初始化模型
model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些假数据
inputs = torch.randn(32, 10)
targets = torch.randn(32, 10)

# 前向传播
outputs, _ = model(inputs)
loss = criterion(outputs, targets)

for name, param in model.named_parameters():
    print(f'{name} updated: {param.requires_grad and param.grad is not None}')
print('===' * 30)

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 检查参数是否更新
for name, param in model.named_parameters():
    print(f'{name} updated: {param.requires_grad and param.grad is not None}')
'''


# import glob
# chars = set()
# files = glob.glob(r'D:\LipData\CMLR\text\**\*.txt', recursive=True)
# with open('zh_vocab.txt', 'w', encoding='utf-8') as f:
#     for fn in files:
#         with open(fn, 'r', encoding='utf-8') as f2:
#             line = f2.readline().strip()
#             chars.update(set(line))
#     f.write('\n'.join(chars))
# print('Done')


import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_boundaries):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries
        self.num_buckets = len(bucket_boundaries) + 1
        self.bucket_sizes = [0] * self.num_buckets
        self.bucket_indices = {i: [] for i in range(self.num_buckets)}

        # 将数据分配到不同的桶中
        for idx, item in enumerate(dataset):
            length = len(item)
            bucket_idx = self._get_bucket_index(length)
            self.bucket_indices[bucket_idx].append(idx)
            self.bucket_sizes[bucket_idx] += 1

    def _get_bucket_index(self, length):
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)

    def __iter__(self):
        # 打乱每个桶中的数据
        for bucket_idx in range(self.num_buckets):
            np.random.shuffle(self.bucket_indices[bucket_idx])
        _batch = []
        for bucket_idx in range(self.num_buckets):
            for idx in self.bucket_indices[bucket_idx]:
                _batch.append(idx)
                if len(_batch) == self.batch_size:
                    yield _batch
                    _batch = []
        if _batch:
            yield _batch

    def __len__(self):
        return sum(self.bucket_sizes)


def collate_fn(batch):
    lengths = [len(item) for item in batch]
    max_length = max(lengths)
    padded_data = torch.zeros(len(batch), max_length, batch[0].size(1))
    lens = []
    for i, item in enumerate(batch):
        lens.append(len(item))
        padded_data[i, :item.size(0)] = item
    print(lens)
    return padded_data


data = [torch.randn(random.randint(10, 100), 10) for _ in range(1000)]
dataset = MyDataset(data)
# 定义分桶边界
bucket_boundaries = [20, 40, 60, 80]
# 定义分桶采样器
bucket_sampler = BucketSampler(dataset, batch_size=10, bucket_boundaries=bucket_boundaries)
data_loader = DataLoader(dataset, batch_sampler=bucket_sampler, collate_fn=collate_fn)

for batch in data_loader:
    print(batch)



# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         return item, len(item)
#
# class BucketBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, bucket_size, shuffle=True, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.bucket_size = bucket_size
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#
#     def create_batches(self):
#         indices = np.argsort([len(item[0]) for item in self.dataset])   # 返回升序排序的索引
#         buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
#
#         if self.shuffle:
#             np.random.shuffle(buckets)
#
#         batches = []
#         for bucket in buckets:
#             if len(bucket) > self.batch_size:
#                 sub_batches = [bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
#                 if self.drop_last and len(sub_batches[-1]) < self.batch_size:
#                     sub_batches = sub_batches[:-1]
#                 batches.extend(sub_batches)
#             else:
#                 if not self.drop_last:
#                     batches.append(bucket)
#
#         if self.shuffle:
#             np.random.shuffle(batches)
#
#         return batches
#
#     def __iter__(self):
#         batches = self.create_batches()
#         for batch in batches:
#             yield batch
#
#     def __len__(self):
#         return len(self.create_batches())
#
#
# def collate_fn(batch):
#     data = [item[0] for item in batch]
#     lengths = [item[1] for item in batch]
#     max_length = max(lengths)
#
#     padded_data = torch.zeros(len(data), max_length, data[0].size(1))
#     for i, item in enumerate(data):
#         padded_data[i, :item.size(0), :] = item
#
#     return padded_data, lengths
#
#
# data = [torch.randn(random.randint(10, 100), 10) for _ in range(1000)]
# dataset = CustomDataset(data)
# bucket_sampler = BucketBatchSampler(dataset, batch_size=10, bucket_size=50, shuffle=True)
# data_loader = DataLoader(dataset, batch_sampler=bucket_sampler, collate_fn=collate_fn)
#
# for batch in data_loader:
#     print(batch)

