# import matplotlib.pyplot as plt
# import numpy as np
# import librosa
# import librosa.display
import torch
import torch.nn.functional as F
from torch import nn


def norm(p1, n1):
    p1 = F.normalize(p1, dim=-1)
    print(p1)
    n1 = F.normalize(n1, dim=-1)
    loss = torch.norm(p1 - n1, dim=-1).mean()
    return loss


def lengths_to_mask(lengths):
    """
    根据序列长度生成对应的掩码。

    参数:
    lengths (torch.Tensor): 序列长度，形状为 (batch_size,)

    返回:
    torch.Tensor: 掩码，形状为 (batch_size, max_length)，值为0或1
    """
    batch_size = lengths.shape[0]
    max_length = torch.max(lengths).item()
    # 生成位置索引 tensor，形状为 (batch_size, max_length)
    pos = torch.arange(max_length, device=lengths.device).unsqueeze(0).expand(batch_size, max_length)
    # 生成长度 tensor，形状为 (batch_size, 1)
    length_expand = lengths.unsqueeze(1)
    # 生成掩码，True表示1，False表示0
    mask = pos < length_expand
    return mask.float()


class UpsampleConv(nn.Module):
    """
    上采样 + 卷积模块
    ``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'``, ``'trilinear',  area``.
    """
    def __init__(self, in_channels, out_channels, scale_factor=(1, 2, 2), mode='area'):
        super(UpsampleConv, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)  # 上采样
        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')  # 卷积
        # self.relu = nn.ReLU(inplace=True)  # 激活函数
        self.dec = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), padding_mode='replicate'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear'),
            nn.Conv3d(32, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), padding_mode='replicate'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear'),
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), padding_mode='replicate'),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear'),
            nn.Conv3d(16, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0), padding_mode='replicate'),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.upsample(x)  # 上采样
        # x = self.conv(x)      # 卷积
        # x = self.relu(x)      # 激活
        return self.dec(x)


if __name__ == '__main__':
    # # 读取音频文件
    # file_path = r'D:\LipData\GRID\audio\s1\bwwbzp.wav'  # 替换为你的音频文件路径
    # y, sr = librosa.load(file_path, sr=None)  # sr=None 保持原始采样率
    #
    # # 绘制音频波形
    # plt.figure(figsize=(14, 5))
    # librosa.display.waveshow(y, sr=sr)
    # plt.title('Audio Waveform')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.show()

    # x = torch.randn(3, 5)
    # y = torch.randn(3, 5)
    # print(x)
    # l = norm(x, y)
    # print(x)
    # print(l)

    # x = torch.randn(16, 1, 32, 32)   # B C H W
    # y = F.interpolate(x, scale_factor=2, mode='bilinear')   # (N, C, d1, d2, ...,dK)  (s1, s2, ...,sK)
    # x = torch.randn(16, 1, 10, 32, 32)   # B C T H W
    # y = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')  # (N, C, d1, d2, ...,dK)  (s1, s2, ...,sK)
    # y = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))(y)
    # print(y.shape)

    # x = torch.randn(3, 5, 6)
    # lens = torch.tensor([5, 3, 4])
    # mask = lengths_to_mask(lens)
    # print(x, mask)
    # print((x * mask.unsqueeze(-1)).sum(dim=1) / lens.unsqueeze(-1).float())

    x = torch.randn((16, 1, 75, 1, 2))   # B C T H W
    y = UpsampleConv(1, 20)(x)
    print(y.shape)