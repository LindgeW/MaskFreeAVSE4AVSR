import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 加载语音文件
file_path = r'D:\LipData\GRID\audio\s1\bwwbzp.wav'
y, sr = librosa.load(file_path, sr=None)

# 计算梅尔频谱图
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# 将梅尔频谱图转换为对数刻度
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
print(log_mel_spectrogram.shape)
# 将梅尔频谱图转换为3D张量
time_steps = log_mel_spectrogram.shape[1]
frequency_bins = log_mel_spectrogram.shape[0]
depth = 16  # 这里假设深度为16，可以根据需要调整

# 创建3D张量
spectrogram_3d = np.expand_dims(log_mel_spectrogram, axis=2)
spectrogram_3d = np.repeat(spectrogram_3d, depth, axis=2)
print(spectrogram_3d.shape)
spectrogram_3d = torch.tensor(spectrogram_3d, dtype=torch.float32).unsqueeze(0)
print(spectrogram_3d.shape)

# 定义3D CNN模型
class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return x


model = AudioFeatureExtractor()
output = model(spectrogram_3d)
print(output.shape)
print("Feature extraction completed.")