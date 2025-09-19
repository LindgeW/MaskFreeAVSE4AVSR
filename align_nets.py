import torch
import torch.nn as nn


class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        """
        mode: the way of aligning
            avg_pool, ada_pool, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ada_pool', 'interpolate', 'conv1d']

        # in_dim_t, in_dim_a, in_dim_v = args.feature_dims
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens
        self.dst_len = seq_len_t
        self.mode = mode

        self.ALIGN_WAY = {
            'ada_pool': self.__ada_pool,
            'avg_pool': self.__avg_pool,
            'interpolate': self.__interpolate,
            'conv1d': self.__conv1d
        }

        if mode == 'conv1d':
            self.conv1d_A = nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_V = nn.Conv1d(seq_len_v, self.dst_len, kernel_size=1, bias=False)
        elif mode == 'ada_pool':
            pass

    def get_seq_len(self):
        return self.dst_len

    def __ada_pool(self, audio_x, video_x):
        audio_x = nn.functional.adaptive_avg_pool1d(audio_x.transpose(1, 2), self.dst_len).transpose(1, 2) if audio_x.size(1) != self.dst_len else audio_x
        video_x = nn.functional.adaptive_avg_pool1d(video_x.transpose(1, 2), self.dst_len).transpose(1, 2) if video_x.size(1) != self.dst_len else video_x
        return audio_x, video_x

    def __interpolate(self, audio_x, video_x):   # down/up samples the input.
        # (B, T, D)
        # mode: linear, bilinear, bicubic, nearest(默认)
        # audio_x = nn.functional.interpolate(audio_x.unsqueeze(1), size=self.dst_len, mode='linear', align_corners=True).squeeze(1) if audio_x.size(1) != self.dst_len else audio_x
        # video_x = nn.functional.interpolate(video_x.unsqueeze(1), size=self.dst_len, mode='linear', align_corners=True).squeeze(1) if video_x.size(1) != self.dst_len else video_x
        audio_x = nn.functional.interpolate(audio_x, size=self.dst_len) if audio_x.size(1) != self.dst_len else audio_x
        video_x = nn.functional.interpolate(video_x, size=self.dst_len) if video_x.size(1) != self.dst_len else video_x
        return audio_x, video_x

    def __avg_pool(self, audio_x, video_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        audio_x = align(audio_x)
        video_x = align(video_x)
        return audio_x, video_x

    def __conv1d(self, audio_x, video_x):
        audio_x = self.conv1d_A(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.conv1d_V(video_x) if video_x.size(1) != self.dst_len else video_x
        return audio_x, video_x

    def forward(self, audio_x, video_x):
        # already aligned
        if audio_x.size(1) == video_x.size(1):
            return audio_x, video_x
        return self.ALIGN_WAY[self.mode](audio_x, video_x)