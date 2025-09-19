import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ctc_decode import *
from multiprocessing import Pool
from conformer import Conformer
#from transformer import TransformerEncoder  # 无位置编码
from transformer2 import TransformerEncoder  # 有位置编码
from transformer_bimodal_decoder import TransformerBiModalDecoderLayer, TransformerBiModalDecoder
import random
from batch_beam_search import beam_decode
from club import *
from fusion_method import *
from dist_loss import *
from file_io import write_numpy_to
from einops import rearrange

'''
def diff_loss(x1, x2):  # (B, D1)  (B, D2)
    #if x1.ndim == 3 and x2.ndim == 3:
    #    x1 = torch.flatten(x1, 0, 1)
    #    x2 = torch.flatten(x2, 0, 1)
    #nx1 = F.normalize(x1 - torch.mean(x1, 0), dim=-1)
    #nx2 = F.normalize(x2 - torch.mean(x2, 0), dim=-1)
    nx1 = F.normalize(x1, dim=-1)
    nx2 = F.normalize(x2, dim=-1)
    return torch.mean(F.relu(1. - torch.norm(nx1-nx2, p=2, dim=-1)) ** 2)
    #return torch.mean(torch.matmul(nx1.transpose(-1, -2), nx2).pow(2))  # C x C   效果差！
    #return torch.mean(torch.abs(F.cosine_similarity(x1-torch.mean(x1, 0), x2-torch.mean(x2, 0), dim=-1)))  # 效果较差 nn.CosineEmbeddingLoss
'''


def get_padding_mask_by_lens(lengths, max_len=None):
    '''
     param:   lengths --- [Batch_size]
     return:  mask --- [Batch_size, max_len]   True for padding
    '''
    bs = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(max_len).unsqueeze(0).expand(bs, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)  # True for padding
    return mask


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes // 16)
            self.conv4 = conv1x1(planes // 16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            out = out * w

        out = out + residual
        out = self.relu(out)
        return out


# ResNet18
class ResNet(nn.Module):
    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.bn = nn.BatchNorm1d(512)
         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, se=self.se)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        #x = self.bn(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300, dropout=0.1, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model).float()   # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:  # (B, L, D)
            return self.dropout(x + self.pe[:, :x.size(1)])
        else:  # (L, B, D)
            return self.dropout(x + self.pe[:, :x.size(0)].transpose(0, 1))


class TransDecoder(nn.Module):
    def __init__(self,
                 n_token,
                 d_model,
                 n_layers=3,
                 n_heads=4,
                 ffn_ratio=4,
                 dropout=0.1,
                 max_len=200):
        super(TransDecoder, self).__init__()
        self.d_model = d_model
        self.scale = d_model ** 0.5
        #self.tok_embedding = nn.Embedding(n_token, d_model)
        self.tok_embedding = nn.Embedding(n_token, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   dim_feedforward=d_model*ffn_ratio,
                                                   nhead=n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_token-1)  # excluding bos

    def get_padding_mask_from_lens(self, lengths, max_len=None):
        '''
         param:   lengths --- [Batch_size]
         return:  mask --- [Batch_size, max_len]
        '''
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)  # True for padding
        return mask

    def forward(self, tgt, src_enc, tgt_lens=None, src_lens=None):
        tgt_embed = self.pos_enc(self.tok_embedding(tgt) * self.scale)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)   # 下三角 (下0上-inf)
        src_padding_mask = self.get_padding_mask_from_lens(src_lens, src_enc.size(1))   # True for masking
        tgt_padding_mask = self.get_padding_mask_from_lens(tgt_lens, tgt.size(1))   # True for masking
        dec_out = self.decoder(tgt_embed,
                               src_enc,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_padding_mask,
                               memory_key_padding_mask=src_padding_mask)
        return self.fc(dec_out)


class TransBiModalDecoder(nn.Module):
    def __init__(self,
                 n_token,
                 d_model,
                 n_layers=3,
                 n_heads=4,
                 ffn_ratio=4,
                 dropout=0.1,
                 max_len=200):
        super(TransBiModalDecoder, self).__init__()
        self.d_model = d_model
        self.scale = d_model ** 0.5
        self.tok_embedding = nn.Embedding(n_token, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        decoder_layer = TransformerBiModalDecoderLayer(d_model=d_model,
                                                       dim_feedforward=d_model*ffn_ratio,
                                                       nhead=n_heads,
                                                       dropout=dropout,
                                                       batch_first=True)
        self.decoder = TransformerBiModalDecoder(decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_token-1)  # excluding bos

    def get_padding_mask_from_lens(self, lengths, max_len=None):
        '''
         param:   lengths --- [Batch_size]
         return:  mask --- [Batch_size, max_len]
        '''
        bs = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(max_len).unsqueeze(0).expand(bs, -1).to(lengths.device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)  # True for padding
        return mask

    def forward(self, tgt, aud_enc, vid_enc, tgt_lens=None, aud_lens=None, vid_lens=None):
        tgt_embed = self.pos_enc(self.tok_embedding(tgt) * self.scale)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)   # 下三角 (下0上-inf)
        aud_padding_mask = self.get_padding_mask_from_lens(aud_lens, aud_enc.size(1))   # True for masking
        vid_padding_mask = self.get_padding_mask_from_lens(vid_lens, vid_enc.size(1))  # True for masking
        tgt_padding_mask = self.get_padding_mask_from_lens(tgt_lens, tgt.size(1))   # True for masking
        dec_out = self.decoder(tgt_embed,
                               aud_enc,
                               vid_enc,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_padding_mask,
                               audio_memory_key_padding_mask=aud_padding_mask,
                               video_memory_key_padding_mask=vid_padding_mask)
        return self.fc(dec_out)


# https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
class DownSample(nn.Module):
    def __init__(self, size, mode='nearest'):
        super(DownSample, self).__init__()
        assert mode in ['nearest', 'bilinear', 'linear', 'bicubic']
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class GEGLU(nn.Module):
    # https://arxiv.org/pdf/2002.05202.pdf
    # GEGLU(x, W, V, b, c) = GELU(xW + b) ⊗ (xV + c)
    # GLU = gated linear units
    def forward(self, x):
        # assert x.shape[-1] % 2 == 0
        x, gates = x.chunk(2, dim=-1)
        # print('in middle of geglu x={}, gates={}'.format(x.shape, gates.shape))
        out = x * F.gelu(gates)
        # out = x * F.relu(gates)
        # out = x * F.silu(gates)
        # out = x * F.sigmoid(gates)
        return out


# Product of Experts
# https://github.com/hyoseok1223/Product-of-Experts-GAN
class PoE(nn.Module):
    def __init__(self):
        super(PoE, self).__init__()

    def forward(self, mu_list, logvar_list, eps=1e-8):
        # mu : N x
        T_sum = 0.
        mu_T_sum = 0.
        for mu, logvar in zip(mu_list, logvar_list):
            var = torch.exp(logvar) + eps
            T = 1. / (var + eps)
            T_sum += T
            mu_T_sum += mu * T
        mu = mu_T_sum / T_sum
        var = 1. / T_sum
        logvar = torch.log(var + eps)
        return mu, logvar


def product_of_experts(mus, logvars):
    # 计算每个专家的精度（precision）
    T = 1.0 / torch.exp(logvars)
    # 计算联合分布的均值和方差
    mu_poe = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
    logvar_poe = torch.log(1.0 / torch.sum(T, dim=0))
    return mu_poe, logvar_poe


# 该模块接收编码后的特征作为输入，通过一个简单的全连接神经网络计算模态质量分数, 使用softmax函数将质量分数转换为选择概率（门控值）
class QualityEstimator(nn.Module):
    """模态质量评估器"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, hidden_dim]
        quality = self.estimator(x)  # [B, T, 1]
        return quality.mean(dim=1)  # [B, 1]


class QualityEstimator(nn.Module):
    """模态质量评估器"""
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, D]
        # 计算时序维度的统计特征
        mean_feat = torch.mean(x, dim=1)  # [B, D]
        std_feat = torch.std(x, dim=1)    # [B, D]
        max_feat = torch.max(x, dim=1)[0]  # [B, D]
        combined = torch.cat([mean_feat, std_feat, max_feat], dim=-1)
        quality = self.estimator(combined)
        return quality  # [B, 1]


class DynamicRouter(nn.Module):
    """动态路由模块"""
    def __init__(self, hidden_dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        '''
        self.router = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            #nn.Softmax(dim=-1)
        )
        '''

        #self.router = nn.Sequential(
        #    nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm1d(hidden_dim),
        #    nn.ReLU(),
        #    nn.Conv1d(hidden_dim, num_experts, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm1d(num_experts),
        #)

        self.router = nn.Linear(hidden_dim*4, num_experts)
        self.dropout = nn.Dropout(0.2)

    def routing(self, v, a):
        ## Linear
        sv = torch.cat([v.mean(dim=1), v.std(dim=1)], dim=-1)
        sa = torch.cat([a.mean(dim=1), a.std(dim=1)], dim=-1)
        logits = self.router(self.dropout(torch.cat([sv, sa], dim=-1)))
        ## Conv
        #cat_av = torch.cat([v, a], dim=-1)   # (B, T, 2*D)
        #logits = self.router(cat_av.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        return logits

    def forward(self, v0, a0, v, a, av, mask=None):
        if mask is None:
            mask = 0.
        #pooled_repr = torch.cat([
        #    v0.mean(dim=1),
        #    a0.mean(dim=1),
        #    #v0.max(dim=1)[0],
        #    #a0.max(dim=1)[0]
        #], dim=-1)   # 可以不avg?
        #rw = F.softmax(self.router(pooled_repr) + mask, dim=-1)  # [B, num_experts]
        
        logits = self.routing(v0, a0)
        #rw = F.softmax(self.router(torch.cat([sv, sa], dim=-1)), dim=-1)
        if self.training:
            #rw = F.gumbel_softmax(self.router(torch.cat([sv, sa], dim=-1)), tau=0.1, hard=True, dim=-1)
            # Straight Through.
            y_soft = F.softmax(logits, dim=-1)
            #y_soft = F.softmax(logits + mask, dim=-1)
            idx = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, idx, 1.0)
            rw = y_hard - y_soft.detach() + y_soft
        else:
            rw = F.one_hot(torch.argmax(logits, dim=-1), self.num_experts).float()
        expert_outputs = torch.stack([v, a, av], dim=1)  # [B, num_experts, T, hidden_dim]
        routing_weights = rw.unsqueeze(-1).unsqueeze(-1)  # [B, num_experts, 1, 1]
        combined_output = (expert_outputs * routing_weights).sum(dim=1)  # [B, T, hidden_dim]
        #return combined_output, routing_weights.squeeze()
        return combined_output, logits


"""序列打分模块"""
class ScoringModule(nn.Module):
    def __init__(self, in_dim, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        dims = [in_dim//3, in_dim//3, in_dim//3]
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_dim, dims[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.Conv1d(dims[0], dims[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(dims[0]),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_dim, dims[1], 5, padding=2, bias=False),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Conv1d(dims[1], dims[1], 5, padding=2, bias=False),
            nn.BatchNorm1d(dims[1]),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_dim, dims[2], 7, padding=3, bias=False),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Conv1d(dims[2], dims[2], 7, padding=3, bias=False),
            nn.BatchNorm1d(dims[2]),
        )
        self.fc = nn.Linear(sum(dims), in_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B, T, D]
    	if not self.batch_first:  # [T, B, D]
    		x = x.transpose(0, 1)
    	res = x  # [B, T, D]
    	x = x.transpose(1, 2)   # [B, D, T]
    	x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
    	x = x.transpose(1, 2)  # [B, T, D]
    	out = res * self.sigmoid(self.fc(x)) + res
    	if not self.batch_first:
    		out = out.transpose(0, 1)   # [T, B, D]
    	return out


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()   # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def attn_pool(tensor, pool, norm=None):
	if pool is None:
		return tensor
	ndim = tensor.ndim
	if ndim == 3:
		tensor = tensor.unsqueeze(1)
	elif ndim == 4:
		pass
	else:
		raise NotImplementedError(f'Unsupported input dim {tensor.shape}')
	B, N, L, C = tensor.shape   # N = num_heads
	tensor = tensor.reshape(B*N, L, C).transpose(-2, -1)   # for 1DConv pooling
	#tensor = tensor.reshape(B*N, H, W, C).permute(0, 3, 1, 2)   # for 2DConv pooling
	tensor = pool(tensor)
	tensor = tensor.reshape(B, N, C, -1).transpose(-2, -1)
	if norm is not None:
		tensor = norm(tensor)
	if ndim == 3:
		tensor = tensor.squeeze(1)
	return tensor


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, dropout=0., bias=True, batch_first=False, residual_pool=True):
        super().__init__()
        self.dim_out = dim_out
        self.dropout = dropout
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim_out * 3, bias=bias)
        self.pool_q = nn.Conv1d(head_dim, head_dim, kernel_size=3, stride=2, padding=1, groups=head_dim, bias=False)
        self.pool_k = nn.Conv1d(head_dim, head_dim, kernel_size=3, stride=2, padding=1, groups=head_dim, bias=False)
        self.pool_v = nn.Conv1d(head_dim, head_dim, kernel_size=3, stride=2, padding=1, groups=head_dim, bias=False)
        self.proj = nn.Linear(dim_out, dim_out)
        self.residual_pool = residual_pool
    	
    def forward(self, x):
    	B, L, _ = x.shape
    	qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # (3, B, H, L, D)
    	q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, L, D)
    	q = attn_pool(q, self.pool_q)  # norm=nn.LayerNorm(head_dim)
    	k = attn_pool(k, self.pool_k)
    	v = attn_pool(v, self.pool_v)
    	attn = (q * self.scale) @ k.transpose(-2, -1)
    	attn_ws = F.softmax(attn, dim=-1)
    	attn_ws = F.dropout(attn_ws, p=self.dropout, training=self.training)
    	x = attn_ws @ v
    	if self.residual_pool:
    		x = x + q
    	x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
    	return self.proj(x)
			
			
class MultiheadPoolAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, dropout=0., bias=True, batch_first=False, residual_pool=True):
        super().__init__()
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, dropout, bias, batch_first, residual_pool)
        self.mlp = nn.Sequential(
            nn.Linear(dim_out, dim_out * 4), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim_out * 4, dim_out), 
            nn.Dropout(dropout) 
        )
        self.pool_skip = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False)   # or None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim_out)
        self.drop_path = DropPath(dropout) if dropout > 0 else nn.Identity()
    	
    def forward(self, x):
    	attn_x = self.attn(self.norm1(x))
    	x_res = attn_pool(x, self.pool_skip)
    	x = x_res + self.drop_path(attn_x)
    	mlp_x = self.mlp(self.norm2(x))
    	x = x + self.drop_path(mlp_x)
    	return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0., batch_first=False, use_residual=True, is_cross_attn=False, has_mlp=True, norm_first=False):
        super().__init__()
        self.use_residual = use_residual
        self.norm_first = norm_first
        self.has_mlp = has_mlp
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=batch_first)
        #self.attn = MultiheadPoolAttention(dim, heads, dropout, batch_first=batch_first)   # from MViT
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim*4, dim), 
            nn.Dropout(dropout) 
        ) if has_mlp else nn.Identity() 
        #self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) if is_cross_attn and norm_first else nn.Identity() 
        self.norm3 = nn.LayerNorm(dim) 
        #self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path = DropPath(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q, kv=None):
        if self.norm_first:
            norm_q, norm_kv = self.norm1(q), self.norm2(kv)
            is_cross_attn = norm_kv is not None
            if is_cross_attn:
                out = self.attn(norm_q, norm_kv, norm_kv, need_weights=False)[0]
            else:
                out = self.attn(norm_q, norm_q, norm_q, need_weights=False)[0]
            x = q + self.drop_path(out) if self.use_residual else out
            return x + self.drop_path(self.mlp(self.norm3(x))) if self.has_mlp else x 
        else:
            is_cross_attn = kv is not None
            if is_cross_attn:
                out = self.attn(q, kv, kv, need_weights=False)[0]
            else:
                out = self.attn(q, q, q, need_weights=False)[0]
            x = self.norm1(q + self.drop_path(out)) if self.use_residual else self.norm1(out)
            return self.norm3(x + self.drop_path(self.mlp(x))) if self.has_mlp else x


class VidTimeSformer(nn.Module):
    def __init__(self, img_w=128, img_h=64, patch_size=16, num_frames=75, in_dim=1, out_dim=512, dropout=0.):
        super(VidTimeSformer, self).__init__()
        num_patches_per_frame = (img_w // patch_size) * (img_h // patch_size)
        self.num_patches_per_frame = num_patches_per_frame
        emb_dim = out_dim // 2
        self.emb_dim = emb_dim
        self.time_emb = nn.Parameter(torch.randn(1, num_frames, emb_dim))
        self.space_emb = nn.Parameter(torch.randn(1, num_patches_per_frame, emb_dim))
        nn.init.trunc_normal_(self.time_emb, 0, 0.02)
        nn.init.trunc_normal_(self.space_emb, 0, 0.02)
        
        #self.patch_to_emb = nn.Conv2d(in_dim, emb_dim, kernel_size=patch_size, stride=patch_size)
        #self.patch_to_emb = nn.Conv3d(in_dim, emb_dim, kernel_size=(3, patch_size, patch_size), stride=(3, patch_size, patch_size))   # tubelet embedding
        self.patch_to_emb = nn.Sequential(nn.Conv2d(in_dim, emb_dim, kernel_size=patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2d(emb_dim))  # work
        self.time_attns = nn.ModuleList([Attention(emb_dim, 4, dropout=dropout, batch_first=True, is_cross_attn=False, has_mlp=False, norm_first=True) for _ in range(3)])
        self.space_attns = nn.ModuleList([Attention(emb_dim, 4, dropout=dropout, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        self.norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim) 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, vid):  # vid: (b, t, c, h, w)
    	b, t, c, h, w = vid.shape
    	x = self.patch_to_emb(vid.reshape(-1, c, h, w)).reshape(b, t, -1, self.emb_dim)   # (B, T, num_patches, D)
        # add spatial embed
    	#x = self.dropout(x + self.space_emb)
        # add time embed
    	#x = self.dropout(x + self.time_emb[:, :t, None, :])
    	for t_attn, s_attn in zip(self.time_attns, self.space_attns):
    		ht = rearrange(x, 'b t p d -> (b p) t d', b=b)
    		ht = self.dropout(ht + self.time_emb[:, :t])
    		ht = t_attn(ht)
    		hs = rearrange(ht, '(b p) t d -> (b t) p d', b=b)
    		hs = self.dropout(hs + self.space_emb) 
    		hs = s_attn(hs)
    		x = rearrange(hs, '(b t) p d -> b t p d', b=b)
    	out = self.norm(x).mean(dim=2) 
    	return self.proj(out)   # (B, T, D)
    

class ViViT(nn.Module):   # Model2
    def __init__(self, img_w=128, img_h=64, patch_size=16, num_frames=75, in_dim=1, out_dim=512, dropout=0.):
        super(ViViT, self).__init__()
        num_patches_per_frame = (img_w // patch_size) * (img_h // patch_size)
        self.num_patches_per_frame = num_patches_per_frame
        emb_dim = out_dim // 2
        self.emb_dim = emb_dim
        self.time_emb = nn.Parameter(torch.randn(1, num_frames, emb_dim))
        self.space_emb = nn.Parameter(torch.randn(1, num_patches_per_frame, emb_dim))
        nn.init.trunc_normal_(self.time_emb, 0, 0.02)
        nn.init.trunc_normal_(self.space_emb, 0, 0.02)
        #self.pos_emb = nn.Parameter(torch.randn(1, num_frames, num_patches_per_frame, emb_dim))
        #nn.init.trunc_normal_(self.pos_emb, 0, 0.02)
        
        #self.patch_to_emb = nn.Conv2d(in_dim, emb_dim, kernel_size=patch_size, stride=patch_size)
        #self.patch_to_emb = nn.Sequential(nn.Conv2d(in_dim, emb_dim, kernel_size=patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2d(emb_dim))  # work
        self.patch_to_emb = nn.Conv3d(in_dim, emb_dim, kernel_size=(5, patch_size, patch_size), stride=(1, patch_size, patch_size), padding=(2, 0, 0))   # tubelet embedding
        
        #self.local_cnn = nn.Sequential(nn.Conv3d(in_dim, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)), nn.ReLU())
        #self.patch_to_emb = nn.Conv2d(32, emb_dim, kernel_size=patch_size, stride=patch_size)

        '''
        self.patch_to_emb = nn.Sequential(  #16x16
            nn.Conv2d(in_dim, emb_dim//4, kernel_size=7, stride=4, padding=3, bias=False), 
            nn.BatchNorm2d(emb_dim//4),
            nn.GELU(),
            #nn.MaxPool2d(kernel_size=7, stride=4, padding=3), 
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(emb_dim//4, emb_dim//4, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(emb_dim//4),
            nn.GELU(),
            nn.Conv2d(emb_dim//4, emb_dim, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(emb_dim),
        )
        self.patch_to_emb = nn.Sequential(  #16x16
            nn.Conv2d(in_dim, emb_dim//4, kernel_size=4, stride=4, bias=False), 
            nn.BatchNorm2d(emb_dim//4),
            nn.GELU(),
            nn.Conv2d(emb_dim//4, emb_dim//4, kernel_size=2, stride=2, bias=False), 
            nn.BatchNorm2d(emb_dim//4),
            nn.GELU(),
            nn.Conv2d(emb_dim//4, emb_dim, kernel_size=2, stride=2, bias=False), 
            nn.BatchNorm2d(emb_dim),
        )
        '''
        self.time_attns = nn.ModuleList([Attention(emb_dim, 4, dropout=dropout, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        self.space_attns = nn.ModuleList([Attention(emb_dim, 4, dropout=dropout, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        self.norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim) 
        self.dropout = nn.Dropout(dropout)
   
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            m.weight.data.normal_(mean=0., std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    
    def forward(self, vid):  # vid: (b, t, c, h, w)
    	b, t, c, h, w = vid.shape
    	#x = self.patch_to_emb(vid.flatten(0, 1)).reshape(b, t, -1, self.emb_dim)   # (B, T, num_patches, D) for 2D
    	x = self.patch_to_emb(vid.transpose(1, 2)).transpose(1, 2).reshape(b, t, -1, self.emb_dim)   # (B, T, num_patches, D) for 3D

    	#vid = self.local_cnn(vid.transpose(1, 2)).transpose(1, 2)   # (B, T, C, H, W)
    	#x = self.patch_to_emb(vid.flatten(0, 1)).reshape(b, t, -1, self.emb_dim)   # (B, T, num_patches, D) 
        
        #x = self.dropout(x + self.pos_emb[:, :t])
    	x = rearrange(x, 'b t p d -> (b t) p d') 
    	x = self.dropout(x + self.space_emb)   # add spatial embed
    	for s_attn in self.space_attns:
    		x = s_attn(x)
    	x = rearrange(x, '(b t) p d -> b t p d', b=b) 
    	x = x.mean(dim=2)   # (b, t, d) 
    	x = self.dropout(x + self.time_emb[:, :t])   # add time embed
    	for t_attn in self.time_attns:
    		x = t_attn(x)
    	out = self.norm(x) 
    	return self.proj(out)   # (B, T, D)


class MWVTP(nn.Module):   # Multi-view Visual Transformer Pooling
    def __init__(self, img_w=128, img_h=64, patch_size=16, num_frames=75, in_dim=1, out_dim=512, dropout=0.1):
        super(MWVTP, self).__init__()
        num_patches_per_frame = (img_w // patch_size) * (img_h // patch_size)
        self.num_patches_per_frame = num_patches_per_frame
        emb_dim = out_dim // 2
        self.emb_dim = emb_dim
        self.patch_to_emb = nn.Sequential(nn.Conv3d(in_dim, 64, kernel_size=5, stride=(1, 2, 2), padding=2, bias=False), 
        								  nn.BatchNorm3d(64), 
        								  nn.ReLU(), 
        								  #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        								  nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False), 
        								  nn.BatchNorm3d(128),
                                          nn.ReLU(),
        								  #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        								  nn.Conv3d(128, emb_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False), 
        								  nn.BatchNorm3d(emb_dim), 
                                        )
        
        #self.roi_size = (img_h//2, img_w//2)   # (h, w)
        #self.roi_size = (48, 88)   # (h, w)
        #num_patches_per_frame = (self.roi_size[0] // 8) * (self.roi_size[1] // 8)
        self.space_emb = nn.Parameter(torch.randn(1, num_patches_per_frame, emb_dim))
        nn.init.trunc_normal_(self.space_emb, 0, 0.02)
        #self.pos_emb = nn.Parameter(torch.randn(1, num_frames, num_patches_per_frame, emb_dim))
        #nn.init.trunc_normal_(self.pos_emb, 0, 0.02)
        
        self.space_attns = nn.ModuleList([Attention(emb_dim, 4, dropout=dropout, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        self.post_norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim) 
        self.dropout = nn.Dropout(dropout)
   
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            m.weight.data.normal_(mean=0., std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    
    def forward(self, vid):  # vid: (b, t, c, h, w)
    	b, t, c, h, w = vid.shape
    	'''	
    	c_y, c_x = h//2, w//2
    	if self.training:
    		offset_y, offset_x = list(torch.randint(-8, 9, (2,)))  # (-8, 8)
    	else:
    		offset_y, offset_x = 0, 0
    	y0 = int(c_y + offset_y - self.roi_size[0]//2)
    	x0 = int(c_x + offset_x - self.roi_size[1]//2)
    	vid = vid[:, :, :, y0: y0+self.roi_size[0], x0: x0+self.roi_size[1]]   # (B, C, T, Rh, Rw)
    	'''
    	x = self.patch_to_emb(vid.transpose(1, 2)).transpose(1, 2).reshape(b, t, -1, self.emb_dim)   # (B, T, num_patches, D) for 3D
    	x = rearrange(x, 'b t p d -> (b t) p d') 
    	x = self.dropout(x + self.space_emb)   # add spatial embed
    	for attn in self.space_attns:
    		x = attn(x)
    	x = self.post_norm(x) 
    	x = rearrange(x, '(b t) p d -> b t p d', b=b) 
    	x = x.mean(dim=2)   # (b, t, d) 
    	return self.proj(x)   # (B, T, D)



class CTCLipModel(nn.Module):
    def __init__(self, vocab_size, se=False):
        super(CTCLipModel, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        #self.vid_enc = VidTimeSformer(128, 64, 16, 80, 1, 512)
        #self.vid_enc = ViViT(128, 64, 16, 80, 1, 512, 0.1)
        #self.vid_enc = MWVTP(128, 64, 16, 80, 1, 512, 0.1)

        self.afront = nn.Sequential(  # 浅层保细节，深层扩感受野
            nn.Conv1d(80, 128, kernel_size=3, stride=1, padding=1, bias=False),  
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  
            nn.BatchNorm1d(512))    
        self.scale = 4
        
        # self.am_embed = nn.Parameter(torch.zeros(512))
        # self.vm_embed = nn.Parameter(torch.zeros(512))
        # 
        # self.av_mlp = nn.Sequential(
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 512*2),
        #     nn.ReLU(),
        #     nn.Linear(512*2, 512))
        # self.ln = nn.LayerNorm(512)

        #self.av_mlp2 = nn.Sequential(
        #    nn.LayerNorm(512),
        #    nn.Linear(512, 512*2),
        #    nn.ReLU(),
        #    #nn.GELU(),
        #    nn.Linear(512*2, 512))

        #self.spk_enc = nn.Sequential(
        #    nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),   
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(),
        #    nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),   
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(),)
        
        # backend: gru/tcn/transformer
        #self.gru = nn.GRU(512, 256, num_layers=3, bidirectional=True, batch_first=True, dropout=0.5)
        self.cfm_a = Conformer(512, 4, 512*4, 3, 31, 0.1)
        self.cfm_v = Conformer(512, 4, 512*4, 3, 31, 0.1)
        #self.cfm_c = Conformer(512, 4, 512*4, 3, 31, 0.1)
        #self.cfm_fusion = Conformer(512, 4, 512*4, 1, 31, 0.1)

        # self.shared_enc = nn.Sequential(
        #          nn.Linear(512, 256),
        #          nn.GELU(),
        #          nn.Linear(256, 256))   # 训练后期，wer难再下降，趋于稳定
        # self.vid_enc = nn.Sequential(
        #          nn.Linear(512, 256),
        #          nn.GELU(),
        #          nn.Linear(256, 256))
        # self.aud_enc = nn.Sequential(
        #          nn.Linear(512, 256),
        #          nn.GELU(),
        #          nn.Linear(256, 256))
        # self.shared_enc = nn.Sequential(nn.Linear(512, 512), GEGLU())  # 较好
        # self.vid_enc = nn.Sequential(nn.Linear(512, 512), GEGLU())
        # self.aud_enc = nn.Sequential(nn.Linear(512, 512), GEGLU())
        #self.shared_enc = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        #self.vid_enc = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        #self.aud_enc = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        # self.shared_enc = Conformer(512, 8, 512*4, 1, 31, 0.1)
        # self.vid_enc = Conformer(512, 8, 512*4, 1, 31, 0.1)
        # self.aud_enc = Conformer(512, 8, 512*4, 1, 31, 0.1)

        # self.recon_v = nn.Linear(512, 512)
        # self.recon_a = nn.Linear(512, 512)

        #latent_dim = 64
        #self.fc_ling_a = nn.Sequential(nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim))
        #self.fc_ling_v = nn.Sequential(nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim))
        
        #self.fc_nonling_a = nn.Linear(512, latent_dim)
        #self.fc_nonling_v = nn.Linear(512, latent_dim)
        # self.fc_ling_av = nn.Sequential(nn.Linear(512, 512), GEGLU())
        # self.fc_ling_a = nn.Sequential(nn.Linear(512, 512), GEGLU(), nn.Linear(256, latent_dim))
        # self.fc_ling_v = nn.Sequential(nn.Linear(512, 512), GEGLU(), nn.Linear(256, latent_dim))
        # self.fc_nonling_a = nn.Sequential(nn.Linear(512, 512), GEGLU())
        # self.fc_nonling_v = nn.Sequential(nn.Linear(512, 512), GEGLU())

        # learnable std
        # self.fc_std_a = nn.Sequential(nn.Linear(512, 256), nn.Tanh())   # nn.Sigmoid()
        # self.fc_std_v = nn.Sequential(nn.Linear(512, 256), nn.Tanh())   # nn.Sigmoid()

        # textual embeddings
        #K, D = 512, 64    # K=128/256/512  D=32/64/128
        #K, D = vocab_size, 256
        #self.txt_emb = nn.Parameter(torch.randn(K, D))
        # self.register_parameter('txt_emb', nn.Parameter(torch.randn(K, 256)))
        #nn.init.xavier_uniform_(self.txt_emb)
        #nn.init.uniform_(self.txt_emb, -1/K, 1/K)
        
        #self.pos_enc = PositionalEncoding(512, batch_first=True) 
        self.fsn = nn.Parameter(torch.randn(4, 512))   # 2, 4, 8, 16
        nn.init.trunc_normal_(self.fsn, 0, 0.02)
        self.a_mbt_attns = nn.ModuleList([Attention(512, 4, 0.1, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        self.v_mbt_attns = nn.ModuleList([Attention(512, 4, 0.1, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        
        #self.fsns = nn.ParameterList([
        #    nn.Parameter(torch.randn(8, 512)),
        #    nn.Parameter(torch.randn(4, 512)),
        #    nn.Parameter(torch.randn(2, 512)),
        #    nn.Parameter(torch.randn(1, 512)),
        #])
        #n_bottleneck = 2 ** (4 - 1)
        #self.fsns = nn.ParameterList([torch.randn(n_bottleneck//(2**i), 512) for i in range(4)])
        #for par in self.fsns: nn.init.trunc_normal_(par, 0, 0.02)

        #self.latents = nn.Parameter(torch.randn(32, 512))
        #nn.init.trunc_normal_(self.latents, 0, 0.02)
        #self.aud_mod_emb = nn.Parameter(torch.randn(1, 1, 512))
        #self.vid_mod_emb = nn.Parameter(torch.randn(1, 1, 512))
        #nn.init.xavier_uniform_(self.aud_mod_emb)
        #nn.init.xavier_uniform_(self.vid_mod_emb)

        # 视听注意力融合
        #self.a_trans = TransformerEncoder(512, 4, 3)
        #self.v_trans = TransformerEncoder(512, 4, 3)
        #self.av_trans = TransformerEncoder(512, 4, 3, attn_mask=True)
        #self.va_trans = TransformerEncoder(512, 4, 3, attn_mask=True)
        #self.a_trans = Conformer(512, 4, 512*4, 3, 31, 0.1)
        #self.v_trans = Conformer(512, 4, 512*4, 3, 31, 0.1)
        #self.av_trans = Conformer(512, 4, 512*4, 3, 31, 0.1)
        
        #self.cross_a = TransformerEncoder(512, 4, 2, attn_mask=False)
        #self.self_a = TransformerEncoder(512, 4, 3, attn_mask=False)
        #self.cross_v = TransformerEncoder(512, 4, 2, attn_mask=False)
        #self.self_v = TransformerEncoder(512, 4, 3, attn_mask=False)
        #self.cross_attns = nn.ModuleList([TransformerEncoder(512, 4, 1, attn_mask=False, need_pos_enc=False) for _ in range(3)])
        #self.self_attns = nn.ModuleList([TransformerEncoder(512, 4, 2, attn_mask=False, need_pos_enc=True) for _ in range(3)])
        
        #self.score_vid = ScoringModule(512, batch_first=False)
        #self.score_aud = ScoringModule(512, batch_first=False)
        
        # for perceiver
        #self.pos_enc = PositionalEncoding(512, batch_first=False) 
        #self.cross_attns = nn.ModuleList([Attention(512, 4, 0.1, batch_first=False, is_cross_attn=True, norm_first=True) for _ in range(2)])
        #self.self_attns = nn.ModuleList([Attention(512, 4, 0.1, batch_first=False, is_cross_attn=False, norm_first=True) for _ in range(4)])
        #self.latent_to_vids = nn.ModuleList([Attention(512, 4, 0.1, batch_first=False, is_cross_attn=True, norm_first=True) for _ in range(2)])
        #self.latent_to_auds = nn.ModuleList([Attention(512, 4, 0.1, batch_first=False, is_cross_attn=True, norm_first=True) for _ in range(2)])
        
        # for mbt
        self.fsn2 = nn.Parameter(torch.randn(4, 512))   # 2, 4, 8, 16
        nn.init.trunc_normal_(self.fsn2, 0, 0.02)
        self.a_mbt_attns2 = nn.ModuleList([Attention(512, 4, 0.1, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        self.v_mbt_attns2 = nn.ModuleList([Attention(512, 4, 0.1, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(3)])
        #self.av_mbt_attns = nn.ModuleList([Attention(512, 4, 0.1, batch_first=True, is_cross_attn=False, norm_first=True) for _ in range(4)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(512) for _ in range(3)])
        
        #self.balance = nn.Parameter(torch.zeros(512))
        #self.sigmoid = nn.Sigmoid()

        #self.ea = nn.Parameter(torch.randn(1, 1, 512))
        #self.ev = nn.Parameter(torch.randn(1, 1, 512))
        #nn.init.orthogonal_(self.ea)
        #nn.init.orthogonal_(self.ev)
        #self.ea = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 512), nn.LayerNorm(512))
        #self.ev = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 512), nn.LayerNorm(512))

        #self.factor = nn.Parameter(torch.zeros(1))  
        #self.q = nn.Parameter(torch.empty(1, 8, 512).normal_(std=0.02))
        #self.autofusion = AutoFusion(512, 512*2)
        #self.cmd_loss = CMD()

        # self.fusion_a = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 256), nn.LayerNorm(256))
        # self.fusion_v = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 256), nn.LayerNorm(256))
        # self.fusion_a = nn.LayerNorm(256)
        # self.fusion_v = nn.LayerNorm(256)
        #self.fusion_av = nn.Linear(512, 256)

        #self.alpha = nn.Parameter(torch.zeros(2))  
        #self.router = DynamicRouter(512)

        #self.aud_decoder = nn.Sequential(
        #    nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(),
        #    nn.ConvTranspose1d(256, 80, kernel_size=4, stride=2, padding=1, bias=False),
        #    nn.BatchNorm1d(80),
            #nn.ReLU(),
            #nn.Conv1d(128, 80, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm1d(80),
        #)
        
        self.aud_decoder = nn.Sequential(
            nn.Conv1d(512, 80 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(80 * 4),   
        #    #nn.PixelShuffle(4),
        )   # (*, Cxr^2, H, W) -> (*, C, Hxr, Wxr)
        
        # ctc
        #self.fc_ctc = nn.Linear(512, vocab_size - 1)  # including blank label, excluding bos
        self.fc_a = nn.Linear(512, vocab_size - 1)  # including blank label, excluding bos
        self.fc_v = nn.Linear(512, vocab_size - 1)  # including blank label, excluding bos

        self.trans_dec = TransDecoder(vocab_size, 512, 3, 4)

        # initialize
        #self._initialize_weights()

    def visual_frontend(self, x):  # (b, t, c, h, w)
        b, t = x.size(0), x.size(1)
        x = self.frontend3D(x.transpose(1, 2)).transpose(1, 2)
        x = torch.flatten(x, 0, 1)
        x = self.resnet18(x)
        return x.reshape(b, t, -1)
    
    def audio_frontend(self, x):  # (b, t, c)
        x = self.afront(x.transpose(1, 2)).transpose(1, 2)
        return x

    def attention(self, q, k, v, mask=None):
        s = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:  # (B, Lq, Lk)
            s = s.masked_fill(mask, -1e9)
        attn = F.softmax(s, dim=-1) @ v
        return attn

    def av_fusion(self, a, v, av_mlp=None):  # (B, Ta, D)  (B, Tv, D)
        av = a + self.attention(a, v, v)  
        #va = v + self.attention(v, a, a) 
        out = self.ln(av + av_mlp(av))
        return out

    def av_fusion_memory(self, a, v, av_mlp=None):  # (B, Ta, D)  (B, Tv, D)
        vq = self.attention(self.q, v, v)     # V -->> q
        av = a + self.factor * self.attention(a, vq, vq)    # q -->> A
        #concat_ = torch.cat((a, v), dim=1)
        #avq = self.attention(self.q, concat_, concat_)    # AV -->> q
        #av = a + self.attention(a, avq, avq)    # q -->> AV
        #va = v + self.attention(v, avq, avq)    # q -->> VA
        out = self.ln(av + av_mlp(av))
        return out

    def orthogonal_consistency_loss(self, a, na, v, nv, alens=None, vlens=None):
        # 一致性约束
        # if alens is not None and vlens is not None:
        #     amask = get_padding_mask_by_lens(alens, a.size(1)).unsqueeze(-1)   # (B, La, 1)
        #     vmask = get_padding_mask_by_lens(vlens, v.size(1)).unsqueeze(-1)   # (B, Lv, 1)
        # else:
        #     amask = vmask = None
        # sim_a = (1. - F.cosine_similarity(self.attention(a, v, v, amask), a, dim=-1)).mean()   # not work well
        # sim_v = (1. - F.cosine_similarity(self.attention(v, a, a, vmask), v, dim=-1)).mean()
        # sim_a = F.mse_loss(self.attention(a, v, v, amask), a)   # work
        # sim_v = F.mse_loss(self.attention(v, a, a, vmask), v)
        # sim_loss = (sim_a + sim_v) / 2.  # 改用对比学习，增强判别能力
        # sim_loss = (1. - F.cosine_similarity(a.mean(1), v.mean(1), dim=-1)).pow(2).mean()
        # sim_loss = 0.01 * barlow_twins_loss(a.mean(dim=1), v.mean(dim=1))
        # sim_loss = 0.01 * in_batch_neg_loss(a.mean(dim=1), v.mean(dim=1))   # in-batch negatives
        sim_loss = 0.01 * (paired_contrastive_loss(a.mean(dim=1), v.mean(dim=1)))
        # sim_loss = 0.01 * (paired_contrastive_loss(a.mean(dim=1), v.mean(dim=1)) +
        #                   (1. - F.cosine_similarity(a.mean(dim=1), v.mean(dim=1), dim=-1).mean()))
        # 正交约束
        # orth_loss = (diff_loss(a, na) + diff_loss(v, nv) + diff_loss(na.mean(1), nv.mean(1))) / 3.
        orth_loss = (hsic(a.reshape(-1, a.size(-1)), na.reshape(-1, na.size(-1))) +
                     hsic(v.reshape(-1, v.size(-1)), nv.reshape(-1, nv.size(-1)))) / 2.
        return sim_loss + orth_loss

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # return mu + eps * std
        return eps.mul(std).add_(mu)

    def resampling(self, mu, std=1.):  # std为常数
        # 并非直接在定义的正态分布上采样，而是先对标准正态分布N(0, 1)进行采样，然后输出`mean + std x 采样值`
        eps = torch.randn_like(mu)
        return mu + eps * std  # 引入高斯噪声N(0, \sigma^2 * I)

    def kl_div(self, mu, logvar):
        # return -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        kl_div = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()  # average over B and T
        return kl_div

    def vib(self, enc_feat):
        mu, std = self.fc_mu(enc_feat), self.fc_std(enc_feat)
        std = F.softplus(std-5, beta=1)
        z = mu + std * torch.randn_like(std)
        kl_div = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(-1).mean()
        return z, kl_div

    def gumbel_softmax(self, logits, dim=-1, temp=1.0, eps=1e-10):
        """
        Gumbel-Softmax重参数化技巧
        """
        # 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        # 应用温度系数
        soft_samples = F.softmax(
            (logits + gumbel_noise) / temp,
            dim=dim
        )
        return soft_samples


    def forward(self, snr, vid_inp, aud_inp, clean_aud_inp, tgts, vid_lens=None, aud_lens=None, tgt_lens=None):  # (b, t, c, h, w)   (b, t, c)
        '''
        p = torch.rand(1).item()
        if p < 0.1:
            vid_inp = vid_inp.new_zeros(vid_inp.shape)
            #gating_labels = torch.tensor([1] * len(tgts)).to(vid_inp.device)
        elif p < 0.2:
            aud_inp = aud_inp.new_zeros(aud_inp.shape)
            #gating_labels = torch.tensor([0] * len(tgts)).to(aud_inp.device)
        else:
            pass
            #gating_labels = None
        '''
        bs = vid_inp.shape[0]
        vid = self.visual_frontend(vid_inp)
        #vid = self.vid_enc(vid_inp)
        aud = self.audio_frontend(aud_inp)
        # max_aud_len = aud.shape[1]
        # sampling_factor = int(max_aud_len / aud.shape[1] + 0.5)
        # aud_lens = (aud_lens.float() / sampling_factor).ceil().long()
        # aud_lens = torch.clamp(aud_lens, min=0, max=aud.shape[1])
        aud_lens = (aud_lens + self.scale - 1) // self.scale  # time subsampling after CNN striding

        #inp_feat = torch.cat((aud+self.am_embed, vid+self.vm_embed), dim=1)
        #enc_src = self.cfm(inp_feat, src_lens+vid_lens)[:, :aud.shape[1]]
        enc_v, enc_a = self.cfm_v(vid, vid_lens), self.cfm_a(aud, aud_lens)
       
        # 根据SNR，自适应选择不同的bottleneck或subnet
        if snr < 0:
            tgt_idx = 0
        elif snr < 10:
            tgt_idx = 1
        else:
            tgt_idx = 2
        logits = self.router(aud)
        if self.training:
            # Straight Through.
            y_soft = F.softmax(logits, dim=-1)
            #y_soft = F.softmax(logits + mask, dim=-1)
            idx = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, idx, 1.0)
            rw = y_hard - y_soft.detach() + y_soft
        else:
            rw = F.one_hot(torch.argmax(logits, dim=-1), self.num_expert).float()
        expert_outputs = torch.stack([v, a, av], dim=1)  # [B, num_experts, T, hidden_dim]
        routing_weights = rw.unsqueeze(-1).unsqueeze(-1)  # [B, num_experts, 1, 1]
        combined_output = (expert_outputs * routing_weights).sum(dim=1)  # [B, T, hidden_dim]

        # clean audio reconstruction
        fsn = self.fsn.unsqueeze(0).repeat(bs, 1, 1)   # (B, N, D) 
        L = fsn.shape[1] 
        #aud, vid = self.pos_enc(aud), self.pos_enc(vid)   
        for i, (v_mbt, a_mbt) in enumerate(zip(self.v_mbt_attns, self.a_mbt_attns)):
            fsn_v = v_mbt(torch.cat([fsn, enc_v], dim=1))
            fsn, enc_v = fsn_v[:, :L], fsn_v[:, L:]
            fsn_a = a_mbt(torch.cat([fsn, enc_a], dim=1))
            fsn, enc_a = fsn_a[:, :L], fsn_a[:, L:]
        
        ra = self.post_norms[0](enc_a)   # (B, T, D)
        #recon_a = self.aud_decoder(ra.transpose(1, 2)).transpose(1, 2)
        recon_aud = self.aud_decoder(ra.transpose(1, 2)).reshape(ra.shape[0], -1, 4, ra.shape[1]).flatten(2).transpose(1, 2)
        drl_loss = 0.5 * F.l1_loss(recon_aud, clean_aud_inp.detach()) + 0.5 * F.mse_loss(self.audio_frontend(recon_aud), self.audio_frontend(clean_aud_inp))

        #enc_v, enc_a = self.cfm_v(vid, vid_lens).transpose(0, 1), self.cfm_a(aud, aud_lens).transpose(0, 1)
        #enc_v, enc_a = self.cfm_v(vid, vid_lens), self.cfm_a(aud, aud_lens)
        
        #src_lens = (tgts > 0).sum(dim=1)
        #src_lens = torch.tensor([self.latents.shape[0]] * vid.shape[0], device=vid.device)
        #src_lens = aud_lens

        # Time-Concat
        #enc_av = self.cfm_c(torch.cat([enc_a, enc_v], dim=1), enc_a.shape[1]+vid_lens)
        #av = enc_av[:, :enc_a.shape[1]]

        # MBT
        fsn2 = self.fsn2.unsqueeze(0).repeat(bs, 1, 1)   # (B, N, D) 
        L = fsn2.shape[1]
        for i, (v_mbt, a_mbt) in enumerate(zip(self.v_mbt_attns2, self.a_mbt_attns2)):
            fsn_v = v_mbt(torch.cat([fsn2, enc_v], dim=1))
            fsn2, enc_v = fsn_v[:, :L], fsn_v[:, L:]
            fsn_a = a_mbt(torch.cat([fsn2, enc_a], dim=1))
            fsn2, enc_a = fsn_a[:, :L], fsn_a[:, L:]
            
            #v_fsn = v_mbt(torch.cat([enc_v, fsn], dim=0))
            #enc_v, fsn = v_fsn[:-fsn.shape[0]], v_fsn[-fsn.shape[0]:]
            #a_fsn = a_mbt(torch.cat([enc_a, fsn], dim=0))
            #enc_a, fsn = a_fsn[:-fsn.shape[0]], a_fsn[-fsn.shape[0]:]
            
        #for fsn, v_mbt, a_mbt in zip(self.fsns, self.v_mbt_attns, self.a_mbt_attns):
        #    fsn = fsn.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)   # (B, N, D) -> (N, B, D)
        #    fsn_v = v_mbt(torch.cat([fsn, enc_v], dim=0))
        #    fsn, enc_v = fsn_v[:fsn.shape[0]], fsn_v[fsn.shape[0]:]
        #    fsn_a = a_mbt(torch.cat([fsn, enc_a], dim=0))
        #    fsn, enc_a = fsn_a[:fsn.shape[0]], fsn_a[fsn.shape[0]:]
        
        av = self.post_norms[1](enc_a)   # (B, T, D)
        va = self.post_norms[2](enc_v)
        #w = self.sigmoid(self.balance)
        #av = w * av + (1 - w) * va
        
        
        '''
        # Perceiver-MBT
        latents = self.latents.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)   # (B, N, D) -> (N, B, D)
        enc_a, enc_v = self.pos_enc(enc_a) + self.aud_mod_emb, self.pos_enc(enc_v) + self.vid_mod_emb   
        for i, cross_attn in enumerate(self.cross_attns):
            x = torch.cat([enc_a, enc_v], dim=0)   # (Ta+Tv, B, D)
            latents = cross_attn(latents, x)
            for self_attn in self.self_attns:
                latents = self_attn(latents)
            enc_a = self.latent_to_auds[i](enc_a, latents)
            enc_v = self.latent_to_vids[i](enc_v, latents)
        av = self.post_norm(latents).transpose(0, 1)
        '''

        # z_v, non_z_v = self.resampling(self.fc_ling_v(enc_v), 0.), self.resampling(self.fc_nonling_v(enc_v), 0.)
        # z_a, non_z_a = self.resampling(self.fc_ling_a(enc_a), 0.), self.resampling(self.fc_nonling_a(enc_a), 0.)
        # z_v, non_z_v = self.resampling(self.fc_ling_av(enc_v), 0.), self.resampling(self.fc_nonling_v(enc_v), 0.)
        # z_a, non_z_a = self.resampling(self.fc_ling_av(enc_a), 0.), self.resampling(self.fc_nonling_a(enc_a), 0.)
        # z_v, non_z_v = self.resampling(self.fc_ling_v(enc_v), 1.), self.resampling(self.fc_nonling_v(enc_v), 0.)
        # z_a, non_z_a = self.resampling(self.fc_ling_a(enc_a), 1.), self.resampling(self.fc_nonling_a(enc_a), 0.)
        #z_v, z_a = self.fc_ling_v(enc_v), self.fc_ling_a(enc_a)

        #if z_v.shape[1] > z_a.shape[1]:  # downsampling
        #    z_v = F.adaptive_avg_pool1d(z_v.transpose(1, 2), z_a.shape[1]).transpose(1, 2)
        #    src_lens = aud_lens
        #elif z_a.shape[1] > z_v.shape[1]:
        #    z_a = F.adaptive_avg_pool1d(z_a.transpose(1, 2), z_v.shape[1]).transpose(1, 2)
        #    src_lens = vid_lens
        #else:
        #    src_lens = vid_lens

        # 引入文本信息
        # ctc_aout, ctc_vout = self.fc_ctc(z_a), self.fc_ctc(z_v)
        # aud_txt_emb = torch.matmul(F.softmax(ctc_aout, dim=-1), self.txt_emb)
        # vid_txt_emb = torch.matmul(F.softmax(ctc_vout, dim=-1), self.txt_emb)

        # 内积相似度来确定权重
        # e_v = F.softmax(z_v @ self.txt_emb.T, dim=-1) @ self.txt_emb
        # e_a = F.softmax(z_a @ self.txt_emb.T, dim=-1) @ self.txt_emb
        # 负欧式距离来确定加权权重 (引入Gumbel噪声，softmax分布趋近于硬量化的离散分布)
        #logits_v = -torch.cdist(z_v, self.txt_emb)
        #logits_a = -torch.cdist(z_a, self.txt_emb)
        # logits_v = -torch.sqrt(torch.sum((z_v.unsqueeze(2) - self.txt_emb.unsqueeze(0)) ** 2, dim=-1))
        # logits_a = -torch.sqrt(torch.sum((z_a.unsqueeze(2) - self.txt_emb.unsqueeze(0)) ** 2, dim=-1))
        # z_v = self.gumbel_softmax(logits_v) @ self.txt_emb
        # z_a = self.gumbel_softmax(logits_a) @ self.txt_emb
        # z_v = F.gumbel_softmax(logits_v, dim=-1, hard=True) @ self.txt_emb
        # z_a = F.gumbel_softmax(logits_a, dim=-1, hard=True) @ self.txt_emb
        #p_v = F.softmax(logits_v, dim=-1) 
        #p_a = F.softmax(logits_a, dim=-1)
        #e_v = p_v @ self.txt_emb
        #e_a = p_a @ self.txt_emb
        #x_v = self.fc_v(z_v + e_v)
        #x_a = self.fc_a(z_a + e_a)

        # enc_vp, enc_ap = self.vid_enc(enc_v), self.aud_enc(enc_a)
        # enc_vc, enc_ac = self.shared_enc(enc_v), self.shared_enc(enc_a)
        #enc_vp, enc_ap = self.vid_enc(enc_v.transpose(1, 2)).transpose(1, 2), self.aud_enc(enc_a.transpose(1, 2)).transpose(1, 2)
        #enc_vc, enc_ac = self.shared_enc(enc_v.transpose(1, 2)).transpose(1, 2), self.shared_enc(enc_a.transpose(1, 2)).transpose(1, 2)
        # enc_vp, enc_ap = self.vid_enc(enc_v, vid_lens), self.aud_enc(enc_a, aud_lens)
        # enc_vc, enc_ac = self.shared_enc(enc_v, vid_lens), self.shared_enc(enc_a, aud_lens)

        # 模态正交与一致性约束
        # orth_loss = diff_loss(enc_vp, enc_vc) + diff_loss(enc_ap, enc_ac) + diff_loss(enc_vp.mean(1), enc_ap.mean(1))
        # sim_loss = (1. - F.cosine_similarity(enc_vc.mean(1), enc_ac.mean(1), dim=-1)).pow(2).mean()
        # orth_sim_loss = ranking_loss(z_a.mean(1, keepdims=True), non_z_a, z_v.mean(1, keepdims=True), non_z_v, mode='contrastive')   # 效果一般
        # orth_sim_loss = ranking_loss(aud_txt_emb.mean(1, keepdims=True), non_z_a, vid_txt_emb.mean(1, keepdims=True), non_z_v, mode='contrastive')
        # orth_sim_loss = self.orthogonal_consistency_loss(z_a, non_z_a, z_v, non_z_v, aud_lens, vid_lens)
        #drl_loss = 0.02 * (paired_contrastive_loss(e_v.mean(dim=1), e_a.mean(dim=1)))
        #drl_loss = 0.01 * CMCM_loss(p_v.mean(dim=1), p_a.mean(dim=1))

        # 模态重构
        # recon_v = self.recon_v(torch.cat((z_v, non_z_v), dim=-1))
        # recon_a = self.recon_a(torch.cat((z_a, non_z_a), dim=-1))
        # recon_v = self.recon_v(torch.cat((vid_txt_emb, non_z_v), dim=-1))
        # recon_a = self.recon_a(torch.cat((aud_txt_emb, non_z_a), dim=-1))
        # 模态重构：交换语义
        #recon_v = self.recon_v(torch.cat((enc_vp, enc_ac), dim=-1))
        #recon_a = self.recon_a(torch.cat((enc_ap, enc_vc), dim=-1))
        # recon_loss = (F.mse_loss(recon_v, enc_v) + F.mse_loss(recon_a, enc_a)) / 2.

        # 重构结果循环送入
        #enc_v_cyc, enc_a_cyc = self.vid_enc(recon_v.transpose(1, 2)).transpose(1, 2), self.aud_enc(recon_a.transpose(1, 2)).transpose(1, 2)
        # enc_v_cyc, enc_a_cyc = self.vid_enc(recon_v, vid_lens), self.aud_enc(recon_a, aud_lens)
        # cyc_loss = (F.mse_loss(self.resampling(self.fc_nonling_v(recon_v), 0.), non_z_v) +
        #             F.mse_loss(self.resampling(self.fc_nonling_a(recon_a), 0.), non_z_a)) / 2.
       
        #av_align = self.av_trans(x_a.transpose(0, 1), x_v.transpose(0, 1), x_v.transpose(0, 1)).transpose(0, 1)
        #va_align = self.va_trans(x_v.transpose(0, 1), x_a.transpose(0, 1), x_a.transpose(0, 1)).transpose(0, 1)
        #a2v = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
        #v2a = self.va_trans(enc_v.transpose(0, 1), enc_a.transpose(0, 1), enc_a.transpose(0, 1)).transpose(0, 1)
        #a = self.av_trans(enc_a.transpose(0, 1)).transpose(0, 1)
        #v = self.av_trans(enc_v.transpose(0, 1)).transpose(0, 1)
        
        #enc_src = self.av_fusion(enc_ac, enc_vc, self.av_mlp)
        #enc_src = torch.cat((enc_ac+self.am_embed, enc_vc+self.vm_embed), dim=1)
        #enc_src = self.cfm_fusion(enc_src, src_lens+vid_lens)[:, :aud.shape[1]]
        #enc_src = self.cfm_fusion(enc_src, aud.shape[1]+vid_lens)[:, :aud.shape[1]]

        # p = torch.rand(1).item()
        # if p < 0.25:
        #     w_a, w_v = 1, 0
        # elif p < 0.5:
        #     w_a, w_v = 0, 1
        # else:
        #     w_a, w_v = 0.5, 0.5
        # av = w_a * self.fusion_a(z_a) + w_v * self.fusion_v(z_v)
        # av = self.fusion_a(z_a) + self.fusion_v(z_v)
        #av = self.cfm_c(self.fusion_av(torch.cat((x_v, x_a), dim=-1)), src_lens)
        #av = self.cfm_c(self.fusion_av(torch.cat((av_align, va_align), dim=-1)), src_lens)
        '''
        p = torch.rand(1).item()
        if p < 0.2:  # drop video
            #vid = enc_v = enc_a.new_zeros(enc_a.shape)
            enc_v = enc_a.new_zeros(enc_a.shape)
            mask = enc_a.new_full((enc_a.shape[0], 3), float('-inf'))
            mask[:, 1] = 0.
        elif p < 0.4:  # drop audio
            #aud = enc_a = enc_v.new_zeros(enc_v.shape)
            enc_a = enc_v.new_zeros(enc_v.shape)
            mask = enc_v.new_full((enc_v.shape[0], 3), float('-inf'))
            mask[:, 0] = 0.
        else:
            mask = None
        '''

        #ea = self.ea(enc_a.mean(1, keepdims=True))
        #ev = self.ev(enc_v.mean(1, keepdims=True))
        #ea, ev = self.ea, self.ev
        #ha = self.av_trans((enc_a + ea).transpose(0, 1)).transpose(0, 1)
        #hv = self.av_trans((enc_v + ev).transpose(0, 1)).transpose(0, 1)
        #a2v = self.av_trans((enc_a + ea).transpose(0, 1), (enc_v + ev).transpose(0, 1), (enc_v + ev).transpose(0, 1)).transpose(0, 1)
        #v2a = self.av_trans((enc_v + ev).transpose(0, 1), (enc_a + ea).transpose(0, 1), (enc_a + ea).transpose(0, 1)).transpose(0, 1)
        #fw = torch.softmax(self.alpha, dim=0)
        #hav = fw[0] * a2v + fw[1] * v2a
        #hav = 0.5 * a2v + 0.5 * v2a  # better
        #a2v = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
        #v2a = self.va_trans(enc_v.transpose(0, 1), enc_a.transpose(0, 1), enc_a.transpose(0, 1)).transpose(0, 1)
        #ha = self.a_trans(enc_a, aud_lens)
        #hv = self.v_trans(enc_v, vid_lens)
        #a2v, v2a = self.av_trans(torch.cat((enc_a, enc_v), dim=1), enc_a.size(1)+vid_lens).split([enc_a.size(1), enc_v.size(1)], dim=1)
        #av, gating_logits = self.router(enc_v.detach(), enc_a.detach(), hv, ha, hav)
        #av = self.router(enc_v, enc_a, enc_v, enc_a, v2a, a2v, mask)[0]
        #av = self.cfm_c(av, src_lens)
       
        #if gating_labels is None:
        #    ctc_log_probs_v = self.fc_ctc(l2norm(v)).log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        #    ctc_log_probs_a = self.fc_ctc(l2norm(a)).log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        #    ctc_log_probs_av = self.fc_ctc(l2norm(f_av)).log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        #    ctc_loss_v = F.ctc_loss(ctc_log_probs_v, tgts[:, 1:], src_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True, reduction='none')
        #    ctc_loss_a = F.ctc_loss(ctc_log_probs_a, tgts[:, 1:], src_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True, reduction='none')
        #    ctc_loss_av = F.ctc_loss(ctc_log_probs_av, tgts[:, 1:], src_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True, reduction='none')
        #    gating_labels = torch.stack([ctc_loss_v, ctc_loss_a, ctc_loss_av]).detach().argmin(dim=0)
        #drl_loss = F.cross_entropy(gating_logits, gating_labels, label_smoothing=0)
        
        # work !!
       #orth_sim_loss = ((diff_loss(z_a, non_z_a) + diff_loss(z_v, non_z_v) + diff_loss(non_z_a.mean(dim=1), non_z_v.mean(dim=1))) / 3. +
        #                 (F.kl_div(z_a.reshape(-1, z_a.size(-1)).log_softmax(dim=-1), av_align.detach().reshape(-1, av_align.size(-1)).softmax(dim=-1), reduction='batchmean') +
        #                  F.kl_div(z_v.reshape(-1, z_v.size(-1)).log_softmax(dim=-1), va_align.detach().reshape(-1, va_align.size(-1)).softmax(dim=-1), reduction='batchmean')) / 2.)
        # 效果都一般
        # orth_sim_loss = ((diff_loss(z_a, non_z_a) + diff_loss(z_v, non_z_v)) / 2.
        #                  + (F.mse_loss(z_a, av_align.detach()) + F.mse_loss(z_v, va_align.detach())) / 2.)
        # orth_sim_loss = ((hsic(z_a.reshape(-1, z_a.size(-1)), non_z_a.reshape(-1, non_z_a.size(-1))) +
        #                   hsic(z_v.reshape(-1, z_v.size(-1)), non_z_v.reshape(-1, non_z_v.size(-1)))) / 2.
        #                  + 0.1 * (F.mse_loss(z_a, av_align.detach()) + F.mse_loss(z_v, va_align.detach())) / 2.)

        # drl_loss = orth_loss + sim_loss + recon_loss + cyc_loss
        # drl_loss = orth_sim_loss + recon_loss + cyc_loss
        
        ctc_loss = F.ctc_loss(self.fc_a(av).log_softmax(dim=-1).transpose(0, 1), tgts[:, 1:], aud_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True) + F.ctc_loss(self.fc_v(va).log_softmax(dim=-1).transpose(0, 1), tgts[:, 1:], vid_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True)
        #ctc_log_probs = self.fc_ctc(av).log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        #ctc_loss = F.ctc_loss(ctc_log_probs, tgts[:, 1:], aud_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True)
        dec_out = self.trans_dec(tgts[:, :-1], av, tgt_lens, aud_lens)
        attn_loss = F.cross_entropy(dec_out.transpose(-1, -2), tgts[:, 1:], ignore_index=0)
        avsr_loss = 0.9 * attn_loss + 0.1 * ctc_loss
        #avsr_loss = attn_loss + 0.3 * ctc_loss
        return avsr_loss, drl_loss

    def beam_search_decode(self, vid_inp, aud_inp, vid_lens, aud_lens, bos_id, eos_id, max_dec_len=50, pad_id=0):
        with torch.no_grad():
            bs = vid_inp.shape[0]
            #vid_inp = vid_inp.new_zeros(vid_inp.shape)
            #aud_inp = aud_inp.new_zeros(aud_inp.shape)
            vid = self.visual_frontend(vid_inp)
            #vid = self.vid_enc(vid_inp)
            aud = self.audio_frontend(aud_inp)
            # max_aud_len = aud.shape[1]
            # sampling_factor = int(max_aud_len / aud.shape[1] + 0.5)
            # aud_lens = (aud_lens.float() / sampling_factor).ceil().long()
            # aud_lens = torch.clamp(aud_lens, min=0, max=aud.shape[1])
            aud_lens = (aud_lens + self.scale - 1) // self.scale  # time subsampling after CNN striding
        
            #if vid.shape[1] > aud.shape[1]:  # downsampling
            #    sampling_factor = int(vid.shape[1] / aud.shape[1] + 0.5)
            #    vid = F.adaptive_avg_pool1d(vid.transpose(1, 2), aud.shape[1]).transpose(1, 2)
            #    vid_lens = (vid_lens.float() / sampling_factor).ceil().long()
            #elif aud.shape[1] > vid.shape[1]:
            #    sampling_factor = int(aud.shape[1] / vid.shape[1] + 0.5)
            #    aud = F.adaptive_avg_pool1d(aud.transpose(1, 2), vid.shape[1]).transpose(1, 2)
            #    aud_lens = (aud_lens.float() / sampling_factor).ceil().long()
            #src_lens = torch.max(vid_lens, aud_lens)
            enc_v, enc_a = self.cfm_v(vid, vid_lens), self.cfm_a(aud, aud_lens)
            
            fsn = self.fsn.unsqueeze(0).repeat(bs, 1, 1)   # (B, N, D) 
            L = fsn.shape[1] 
            #aud, vid = self.pos_enc(aud), self.pos_enc(vid)   
            for i, (v_mbt, a_mbt) in enumerate(zip(self.v_mbt_attns, self.a_mbt_attns)):
                fsn_v = v_mbt(torch.cat([fsn, enc_v], dim=1))
                fsn, enc_v = fsn_v[:, :L], fsn_v[:, L:]
                fsn_a = a_mbt(torch.cat([fsn, enc_a], dim=1))
                fsn, enc_a = fsn_a[:, :L], fsn_a[:, L:]

            #enc_v, enc_a = self.cfm_v(vid, vid_lens).transpose(0, 1), self.cfm_a(aud, aud_lens).transpose(0, 1)
            #enc_v, enc_a = self.cfm_v(vid, vid_lens), self.cfm_a(aud, aud_lens)
            #z_v, z_a = self.fc_ling_v(enc_v), self.fc_ling_a(enc_a)
        
            #src_lens = (tgts > 0).sum(dim=1)
            #src_lens = torch.tensor([self.latents.shape[1]] * vid.shape[0], device=vid.device)
            #av = self.cross_v(self.latents.repeat(bs, 1, 1).transpose(0, 1), vid.transpose(0, 1), vid.transpose(0, 1))
            #av = self.self_v(av)
            #av = self.cross_a(av, aud.transpose(0, 1), aud.transpose(0, 1))
            #av = self.self_a(av).transpose(0, 1)
        
            #src_lens = (tgts > 0).sum(dim=1)
            #src_lens = torch.tensor([self.latents.shape[0]] * vid.shape[0], device=vid.device)
            #src_lens = aud_lens
        
            # Time-Concat
            #enc_av = self.cfm_c(torch.cat([enc_a, enc_v], dim=1), enc_a.shape[1]+vid_lens)
            #av = enc_av[:, :enc_a.shape[1]]
            
            # MBT
            fsn2 = self.fsn2.unsqueeze(0).repeat(bs, 1, 1)   # (B, N, D) 
            L = fsn2.shape[1] 
            for i, (v_mbt, a_mbt) in enumerate(zip(self.v_mbt_attns2, self.a_mbt_attns2)):
                fsn_v = v_mbt(torch.cat([fsn2, enc_v], dim=1))
                fsn2, enc_v = fsn_v[:, :L], fsn_v[:, L:]
                fsn_a = a_mbt(torch.cat([fsn2, enc_a], dim=1))
                fsn2, enc_a = fsn_a[:, :L], fsn_a[:, L:]
                
                #enc_v, fsn = v_fsn[:-fsn.shape[0]], v_fsn[-fsn.shape[0]:]
                #a_fsn = a_mbt(torch.cat([enc_a, fsn], dim=0))
                #enc_a, fsn = a_fsn[:-fsn.shape[0]], a_fsn[-fsn.shape[0]:]

            #for fsn, v_mbt, a_mbt in zip(self.fsns, self.v_mbt_attns, self.a_mbt_attns):
            #    fsn = fsn.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)   # (B, N, D) -> (N, B, D)
            #    fsn_v = v_mbt(torch.cat([fsn, enc_v], dim=0))
            #    fsn, enc_v = fsn_v[:fsn.shape[0]], fsn_v[fsn.shape[0]:]
            #    fsn_a = a_mbt(torch.cat([fsn, enc_a], dim=0))
            #    fsn, enc_a = fsn_a[:fsn.shape[0]], fsn_a[fsn.shape[0]:]
            
            av = self.post_norms[1](enc_a)
            #va = self.post_norms[2](enc_v)
            #w = self.sigmoid(self.balance)
            #av = w * av + (1 - w) * va

            '''
            # Perceiver-MBT
            latents = self.latents.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)   # (B, N, D) -> (N, B, D)
            enc_a, enc_v = self.pos_enc(enc_a) + self.aud_mod_emb, self.pos_enc(enc_v) + self.vid_mod_emb   
            for i, cross_attn in enumerate(self.cross_attns):
                x = torch.cat([enc_a, enc_v], dim=0)   # (Ta+Tv, B, D)
                latents = cross_attn(latents, x)
                for self_attn in self.self_attns:
                    latents = self_attn(latents)
                enc_a = self.latent_to_auds[i](enc_a, latents)
                enc_v = self.latent_to_vids[i](enc_v, latents)
            av = self.post_norm(latents).transpose(0, 1)
            '''

            #if z_v.shape[1] > z_a.shape[1]:  # downsampling
            #    z_v = F.adaptive_avg_pool1d(z_v.transpose(1, 2), z_a.shape[1]).transpose(1, 2)
            #    src_lens = aud_lens
            #elif z_a.shape[1] > z_v.shape[1]:
            #    z_a = F.adaptive_avg_pool1d(z_a.transpose(1, 2), z_v.shape[1]).transpose(1, 2)
            #    src_lens = vid_lens
            #else:
            #    src_lens = vid_lens

            # z_v = z_v + F.softmax(z_v @ self.txt_emb.T, dim=-1) @ self.txt_emb
            # z_a = z_a + F.softmax(z_a @ self.txt_emb.T, dim=-1) @ self.txt_emb
            #logits_v = -torch.cdist(z_v, self.txt_emb)   # (B, T, K)
            #logits_a = -torch.cdist(z_a, self.txt_emb)
            # logits_v = -torch.sqrt(torch.sum((z_v.unsqueeze(2) - self.txt_emb.unsqueeze(0)) ** 2, dim=-1))
            # logits_a = -torch.sqrt(torch.sum((z_a.unsqueeze(2) - self.txt_emb.unsqueeze(0)) ** 2, dim=-1))
            #x_v = self.fc_v(z_v + F.softmax(logits_v, dim=-1) @ self.txt_emb)
            #x_a = self.fc_a(z_a + F.softmax(logits_a, dim=-1) @ self.txt_emb)
            # z_v = torch.zeros_like(logits_v).scatter_(-1, torch.argmax(logits_v, dim=-1, keepdim=True), 1) @ self.txt_emb
            # z_a = torch.zeros_like(logits_a).scatter_(-1, torch.argmax(logits_a, dim=-1, keepdim=True), 1) @ self.txt_emb
            # z_v = F.one_hot(torch.argmax(logits_v, dim=-1), num_classes=logits_v.size(-1)).float() @ self.txt_emb
            # z_v = F.one_hot(torch.argmax(logits_a, dim=-1), num_classes=logits_a.size(-1)).float() @ self.txt_emb

            # enc_vp, enc_ap = self.vid_enc(enc_v), self.aud_enc(enc_a)
            # enc_vc, enc_ac = self.shared_enc(enc_v), self.shared_enc(enc_a)
            #enc_vc, enc_ac = self.shared_enc(enc_v.transpose(1, 2)).transpose(1, 2), self.shared_enc(enc_a.transpose(1, 2)).transpose(1, 2)
            # enc_vc, enc_ac = self.shared_enc(enc_v, vid_lens), self.shared_enc(enc_a, aud_lens)

            # write_numpy_to('enc_vp2.txt', enc_vp.mean(dim=1).detach().cpu().numpy())
            # write_numpy_to('enc_ap2.txt', enc_ap.mean(dim=1).detach().cpu().numpy())
            # write_numpy_to('enc_vc2.txt', enc_vc.mean(dim=1).detach().cpu().numpy())
            # write_numpy_to('enc_ac2.txt', enc_ac.mean(dim=1).detach().cpu().numpy())

            #inp_feat = torch.cat((aud+self.am_embed, vid+self.vm_embed), dim=1)
            #enc_src = self.cfm(inp_feat, lens+vid_lens)[:, :aud.shape[1]]
            #enc_v, enc_a = self.cfm_v(vid, vid_lens), self.cfm_a(aud, aud_lens)
            #enc_vc, enc_ac = self.cfm_c(vid, vid_lens), self.cfm_c(aud, aud_lens)

            #av_align = self.av_trans(x_a.transpose(0, 1), x_v.transpose(0, 1), x_v.transpose(0, 1)).transpose(0, 1)
            #va_align = self.va_trans(x_v.transpose(0, 1), x_a.transpose(0, 1), x_a.transpose(0, 1)).transpose(0, 1)
            #av_align = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
            #va_align = self.va_trans(enc_v.transpose(0, 1), enc_a.transpose(0, 1), enc_a.transpose(0, 1)).transpose(0, 1)
            #a = self.av_trans(enc_a.transpose(0, 1)).transpose(0, 1)
            #v = self.av_trans(enc_v.transpose(0, 1)).transpose(0, 1)
            #a2v = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
            #v2a = self.va_trans(enc_v.transpose(0, 1), enc_a.transpose(0, 1), enc_a.transpose(0, 1)).transpose(0, 1)
            
            #enc_src = self.av_fusion(enc_ac, enc_vc, self.av_mlp)
            #enc_src = torch.cat((enc_ac+self.am_embed, enc_vc+self.vm_embed), dim=1)
            #enc_src = self.cfm_fusion(enc_src, lens+vid_lens)[:, :aud.shape[1]]
            #enc_src = self.cfm_fusion(enc_src, aud.shape[1]+vid_lens)[:, :aud.shape[1]]

            # av = self.fusion_a(z_a) + self.fusion_v(z_v)
            #av = self.cfm_c(self.fusion_av(torch.cat((x_v, x_a), dim=-1)), src_lens)
            #av = self.cfm_c(self.fusion_av(torch.cat((av_align, va_align), dim=-1)), src_lens)
            
            ''' 
            idxs = self.router.routing(enc_v, enc_a).argmax(dim=-1)   # (B, )
            av = torch.zeros_like(enc_v)
            for i in range(3):
                if i == 0 and (idxs==0).any():
                    ev = self.ev(enc_v[idxs==i].mean(1, keepdims=True))
                    r = self.av_trans((enc_v[idxs==i] + ev).transpose(0, 1)).transpose(0, 1)
                    av[idxs==i] = r
                elif i == 1 and (idxs==1).any():
                    ea = self.ea(enc_a[idxs==i].mean(1, keepdims=True))
                    r = self.av_trans((enc_a[idxs==i] + ea).transpose(0, 1)).transpose(0, 1)
                    av[idxs==i] = r
                elif i == 2 and (idxs==2).any():
                    ea = self.ea(enc_a[idxs==i].mean(1, keepdims=True))
                    ev = self.ev(enc_v[idxs==i].mean(1, keepdims=True))
                    a2v = self.av_trans((enc_a[idxs==i] + ea).transpose(0, 1), (enc_v[idxs==i] + ev).transpose(0, 1), (enc_v[idxs==i] + ev).transpose(0, 1)).transpose(0, 1)
                    v2a = self.av_trans((enc_v[idxs==i] + ev).transpose(0, 1), (enc_a[idxs==i] + ea).transpose(0, 1), (enc_a[idxs==i] + ea).transpose(0, 1)).transpose(0, 1)
                    r = 0.5 * a2v + 0.5 * v2a   # better
                    av[idxs==i] = r
            '''
            #ea = self.ea(enc_a.mean(1, keepdims=True))
            #ev = self.ev(enc_v.mean(1, keepdims=True))
            #ea, ev = self.ea, self.ev
            #a2v = self.av_trans((enc_a + ea).transpose(0, 1), (enc_v + ev).transpose(0, 1), (enc_v + ev).transpose(0, 1)).transpose(0, 1)
            #v2a = self.av_trans((enc_v + ev).transpose(0, 1), (enc_a + ea).transpose(0, 1), (enc_a + ea).transpose(0, 1)).transpose(0, 1)
            #av = 0.5 * a2v + 0.5 * v2a  # better

            #vr = self.av_trans((enc_v + self.ev).transpose(0, 1)).transpose(0, 1)
            #ar = self.av_trans((enc_a + self.ea).transpose(0, 1)).transpose(0, 1)
            #print(F.cosine_similarity(vr.mean(dim=1), ar.mean(dim=1)))
            '''
            v = self.av_trans((enc_v+self.ev).transpose(0, 1)).transpose(0, 1)
            a = self.av_trans((enc_a+self.ea).transpose(0, 1)).transpose(0, 1)
            a2v = self.av_trans((enc_a+self.ea).transpose(0, 1), (enc_v+self.ev).transpose(0, 1), (enc_v+self.ev).transpose(0, 1)).transpose(0, 1)
            v2a = self.av_trans((enc_v+self.ev).transpose(0, 1), (enc_a+self.ea).transpose(0, 1), (enc_a+self.ea).transpose(0, 1)).transpose(0, 1)
            #a = self.a_trans(enc_a.transpose(0, 1)).transpose(0, 1)
            #v = self.v_trans(enc_v.transpose(0, 1)).transpose(0, 1)
            #a2v = self.a_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
            #v2a = self.v_trans(enc_v.transpose(0, 1), enc_a.transpose(0, 1), enc_a.transpose(0, 1)).transpose(0, 1)
            #fw = torch.softmax(self.alpha, dim=0)
            #f_av = fw[0] * a2v + fw[1] * v2a
            f_av = 0.5 * a2v + 0.5 * v2a   # better
            #a2v = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
            #v2a = self.va_trans(enc_v.transpose(0, 1), enc_a.transpose(0, 1), enc_a.transpose(0, 1)).transpose(0, 1)
            #a = self.a_trans(enc_a, aud_lens)
            #v = self.v_trans(enc_v, vid_lens)
            #a2v, v2a = self.av_trans(torch.cat((enc_a, enc_v), dim=1), enc_a.size(1)+vid_lens).split([enc_a.size(1), enc_v.size(1)], dim=1)
            av = self.router(enc_v, enc_a, v, a, f_av)[0]
            #av = self.router(enc_v, enc_a, enc_v, enc_a, v2a, a2v)[0]
            #av = self.cfm_c(av, src_lens)
            '''
            #res = beam_decode(self.trans_dec, enc_src, src_mask, bos_id, eos_id, max_output_length=max_dec_len, beam_size=10)
            res = beam_decode(self.trans_dec, av, aud_lens, bos_id, eos_id, max_output_length=max_dec_len, beam_size=10)
        return res.detach().cpu()


    def ctc_greedy_decode(self, vids, lens=None):
        with torch.no_grad():
            vid_feat = self.visual_frontend(vids)
            seq_feat = self.cfm(vid_feat, lens)
            logits = self.fc(seq_feat)  # (B, T, V)
            return logits.data.cpu().argmax(dim=-1)

    def ctc_beam_decode(self, vids, lens=None):
        res = []
        with torch.no_grad():
            vid_feat = self.visual_frontend(vids)
            seq_feat = self.cfm(vid_feat, lens)
            logits = self.fc(seq_feat)  # (B, T, V)
            probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            for prob in probs:
                pred = ctc_beam_decode3(prob, 10, 0)
                res.append(pred)
            return res
    
    '''
    def beam_decode(self, vids):
        res = []
        with torch.no_grad():
            logits = self.forward(vids)[0]  # (B, T, V)
            probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            with Pool(len(probs)) as p:
                res = p.map(ctc_beam_decode3, probs)
                #res.append(pred)
            return res
    '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SpeakerIdentity(nn.Module):
    def __init__(self):
        super(SpeakerIdentity, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        #self.pool = nn.AdaptiveAvgPool2d(1)
        #self.proj = nn.Linear(32768, 256)
        #self.gru = nn.GRU(256, 256 // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        #self.head = nn.Sequential(
        #    nn.Linear(32768, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 256))
        self.head = nn.Linear(512, 512)

    def freeze_frontend(self):
        self.frontend3D.requires_grad_(False)
        self.resnet18.requires_grad_(False)
        print('Freeze the weights of visual frontend ...')

    def forward(self, x):  # (b, t, c, h, w)
        b, t = x.shape[:2]
        x = x.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        x = self.frontend3D(x)
        x = x.transpose(1, 2).contiguous()  # (b, t, c, h, w)
        x = x.reshape(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        #x = x.flatten(0, 1)  # (bt, c, h, w)
        #x = self.pool(x)

        x = x.reshape(b, t, -1)
        #feat = self.gru(self.proj(x))[0]  # (b, t, d)
        feat = x.mean(dim=1)
        #feat = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=-1)   # x-vector
        seq_feat = self.head(feat)  # (b, d)
        return x, seq_feat   # (b, d)


# 适用于ASR
class SelfAttentivePooling(nn.Module):
    def __init__(self, hid_size, attn_size=None):
        super(SelfAttentivePooling, self).__init__()
        if attn_size is None:
            attn_size = hid_size // 2
        self.mlp = nn.Sequential(
                nn.Linear(hid_size, attn_size),
                nn.ReLU(),
                nn.Linear(attn_size, 1, bias=False))

    def forward(self, x):  # (B, L, D)
        attn = self.mlp(x).squeeze(2)  # (B, L)
        attn_weights = F.softmax(attn, dim=1)  # (B, L)
        weighted_inputs = torch.mul(x, attn_weights.unsqueeze(2))  # (B, L, D)
        pooled_output = torch.sum(weighted_inputs, dim=1)  # (B, D)
        return pooled_output


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class SpeakerIdentity2D(nn.Module):
    def __init__(self, num_spk):
        super(SpeakerIdentity2D, self).__init__()
        self.frontend2D = nn.Sequential(
            #nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        # self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512))
        #self.head = nn.Linear(512, 512)
        #self.head = SelfAttentivePooling(512, 512//2)
        self.spk_cls = nn.Linear(512, num_spk)

    def forward(self, x):  # (b, t, c, h, w)
        b, t = x.shape[:2]
        x = torch.flatten(x, 0, 1)  # (bt, c, h, w)
        x = self.frontend2D(x)
        x = self.resnet18(x)
        # x = self.pool(x)
        x = x.reshape(b, t, -1)
        # feat = x.mean(dim=1)
        # feat = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=-1)  # i-vector
        #seq_feat = self.head(x)  # (b, d)
        seq_feat = self.head(x.mean(dim=1))  # (b, d)
        return x, seq_feat, self.spk_cls(seq_feat)  # (b, t, d)  (b, d)


class DRLModel(nn.Module):
    def __init__(self, vocab_size):
        super(DRLModel, self).__init__()
        self.avsr = CTCLipModel(vocab_size)
        # self.mi_net = CLUBSample_reshape(512, 512, 512)
        # self.tmp = 1.

    def forward(self, vids, auds, clean_auds, tgts, vid_lens, aud_lens, tgt_lens):   # (B, T, C, H, W)
        avsr_loss, drl_loss = self.avsr(vids, auds, clean_auds, tgts, vid_lens, aud_lens, tgt_lens)
        return {'avsr': avsr_loss, 'drl': drl_loss}
        # return {'avsr': attn_loss, 'drl': drl_loss}

    def load_pretrain_bert(self, text, bert_path):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert_model = BertModel.from_pretrained(bert_path)
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True)
        output = self.bert_model(**encoded_input)
        return output[0][:, 0]

    def calc_triplet_loss(self, vids):
        vids = torch.flatten(vids, 0, 1)
        # vids: (2x16, T, C, H, W)
        frame_feat, seq_feat, _ = self.spk(vids)
        spk1, spk2 = frame_feat.chunk(2, dim=0)   # 不相同  (N, T, D)
        L = random.choice(range(1, spk1.shape[1]))  # [1: T-1]
        spk1_sfl = torch.cat((spk1[:, L:, ...], spk1[:, :L, ...]), dim=1).contiguous()
        spk2_sfl = torch.cat((spk2[:, L:, ...], spk2[:, :L, ...]), dim=1).contiguous()
        # pos: (spk1, spk1_sfl)    neg: (spk2, spk1_sfl)
        # dcl_loss = F.relu(0.2 + F.pairwise_distance(spk1, spk1_sfl, p=2) - F.pairwise_distance(spk2, spk1_sfl, p=2))
        # dcl_loss = F.relu(0.2 + torch.norm(spk1-spk1_sfl, dim=-1, p=2) - torch.norm(spk2-spk1_sfl, dim=-1, p=2))
        # dcl_loss = F.relu(0.2 + torch.cdist(spk1, spk1_sfl, p=2) - torch.cdist(spk2, spk1_sfl, p=2))
        labels = torch.ones(spk1.shape[0]*spk1.shape[1], device=vids.device)
        frame_loss1 = F.cosine_embedding_loss(spk1.flatten(0, 1), spk1_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk2.flatten(0, 1), spk1_sfl.flatten(0, 1), target=-1. * labels, margin=0.2)
        frame_loss2 = F.cosine_embedding_loss(spk2.flatten(0, 1), spk2_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk1.flatten(0, 1), spk2_sfl.flatten(0, 1), target=-1. * labels, margin=0.2)
        frame_loss = 0.5 * frame_loss1 + 0.5 * frame_loss2

        s1, s2 = seq_feat.chunk(2, dim=0)  # 不相同  (N, D)
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        seq_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        seq_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        seq_loss = 0.5 * seq_loss1 + 0.5 * seq_loss2
        return frame_loss + seq_loss

    
    '''
    def calc_triplet_loss(self, vids):
        vids = torch.flatten(vids, 0, 1)
        # vids: (2x16, T, C, H, W)
        spk_feat = self.spk(vids)[0]
        spk1, spk2 = spk_feat.chunk(2, dim=0)   # 不相同  (N, T, D)
        L = random.choice(range(1, spk1.shape[1]))  # [1: T-1]
        spk1_sfl = torch.cat((spk1[:, L:, ...], spk1[:, :L, ...]), dim=1).contiguous()
        spk2_sfl = torch.cat((spk2[:, L:, ...], spk2[:, :L, ...]), dim=1).contiguous()
        # pos: (spk1, spk1_sfl)    neg: (spk2, spk1_sfl)
        # dcl_loss = F.relu(0.2 + F.pairwise_distance(spk1, spk1_sfl, p=2) - F.pairwise_distance(spk2, spk1_sfl, p=2))
        # dcl_loss = F.relu(0.2 + torch.norm(spk1-spk1_sfl, dim=-1, p=2) - torch.norm(spk2-spk1_sfl, dim=-1, p=2))
        # dcl_loss = F.relu(0.2 + torch.cdist(spk1, spk1_sfl, p=2) - torch.cdist(spk2, spk1_sfl, p=2))
        # F.cosine_similarity()
        labels = torch.ones(spk1.shape[0]*spk1.shape[1], device=vids.device)
        loss1 = F.cosine_embedding_loss(spk1.flatten(0, 1), spk1_sfl.flatten(0, 1), target=labels) + F.cosine_embedding_loss(spk2.flatten(0, 1), spk1_sfl.flatten(0, 1), target=-1.*labels, margin=0.2)
        loss2 = F.cosine_embedding_loss(spk2.flatten(0, 1), spk2_sfl.flatten(0, 1), target=labels) + F.cosine_embedding_loss(spk1.flatten(0, 1), spk2_sfl.flatten(0, 1), target=-1. * labels, margin=0.2)
        dcl_loss = loss1 + loss2
        return dcl_loss
    '''

    def calc_orth_loss(self, vids, tgts, spk_ids, xlens, ylens):
        '''
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        '''
        # vids: (2x16, T, C, H, W)
        ## for spk
        _, spk_feat, spk_logits = self.spk(vids)
        spk_loss = F.cross_entropy(spk_logits, spk_ids)
        #s1, s2 = spk_feat.chunk(2, dim=0)  # 不相同  (N, D)
        '''
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        spk_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        spk_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        spk_loss = spk_loss1 + spk_loss2
        '''
        ## for vsr
        #logits, _, cont_feat = self.vsr(vids, xlens)
        #log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        #vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        ctc_logits, dec_logits, _, cont_feat = self.vsr(vids, tgts[:, :-1], xlens, ylens)
        ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        ctc_loss = F.ctc_loss(ctc_log_probs, tgts[:, 1:], xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        attn_loss = F.cross_entropy(dec_logits.transpose(-1, -2).contiguous(), tgts[:, 1:], ignore_index=0)
        vsr_loss = 0.9*attn_loss + 0.1*ctc_loss
        #c1, c2 = cont_feat.chunk(2, dim=0)  # 对应s1, s2  (N, T, D)
        #diff_loss = diff_loss(s1.unsqueeze(1).detach(), c1) + diff_loss(s2.unsqueeze(1).detach(), c2)    # (N, D)  (N, T, D)
        diff_loss = diff_loss(spk_feat.unsqueeze(1).expand_as(cont_feat).detach(), cont_feat) 
        return vsr_loss + spk_loss + diff_loss

    def calc_orth_loss2(self, vids, tgts, spk_ids, xlens, ylens, opt_mi):
        '''
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        '''
        # vids: (2x16, T, C, H, W)
        ## for spk
        _, spk_feat, spk_logits = self.spk(vids)
        spk_loss = F.cross_entropy(spk_logits, spk_ids)
        #s1, s2 = spk_feat.chunk(2, dim=0)  # 不相同  (N, D)
        '''
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        spk_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        spk_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        spk_loss = spk_loss1 + spk_loss2
        '''
        ## for vsr
        #logits, _, cont_feat = self.vsr(vids, xlens)
        #log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        #vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        ctc_logits, dec_logits, _, cont_feat = self.vsr(vids, tgts[:, :-1], xlens, ylens)
        ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        ctc_loss = F.ctc_loss(ctc_log_probs, tgts[:, 1:], xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        attn_loss = F.cross_entropy(dec_logits.transpose(-1, -2).contiguous(), tgts[:, 1:].long(), ignore_index=0)
        vsr_loss = 0.9*attn_loss + 0.1*ctc_loss
        #diff_loss = diff_loss(spk_feat.unsqueeze(1).expand_as(cont_feat).detach(), cont_feat)
        for _ in range(5):  # 将seq-level换成token-level 
            opt_mi.zero_grad()
            #lld_loss = -self.mi_net.loglikeli(spk_feat.detach(), cont_feat.mean(dim=1).detach())
            lld_loss = -self.mi_net.loglikeli(spk_feat.unsqueeze(1).expand_as(cont_feat).detach(), cont_feat.detach())
            lld_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mi_net.parameters(), max_norm=1.)
            opt_mi.step()
        #mi_loss = self.mi_net.mi_est(spk_feat, cont_feat.mean(dim=1))
        mi_loss = self.mi_net.mi_est(spk_feat.unsqueeze(1).expand_as(cont_feat), cont_feat)
        return vsr_loss + 0.5 * spk_loss + 0.01 * mi_loss

    def calc_drl_loss(self, vids, tgts, xlens, ylens):
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        # vids: (2x16, T, C, H, W)
        ## for spk
        frame_feat, seq_feat = self.spk(vids)
        spk1, spk2 = frame_feat.chunk(2, dim=0)  # 不相同  (N, T, D)
        L = random.choice(range(1, spk1.shape[1]))  # [1: T-1]
        spk1_sfl = torch.cat((spk1[:, L:, ...], spk1[:, :L, ...]), dim=1).contiguous()
        spk2_sfl = torch.cat((spk2[:, L:, ...], spk2[:, :L, ...]), dim=1).contiguous()
        labels = torch.ones(spk1.shape[0] * spk1.shape[1], device=vids.device)
        frame_loss1 = F.cosine_embedding_loss(spk1.flatten(0, 1), spk1_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk2.flatten(0, 1), spk1_sfl.flatten(0, 1), target=-1. * labels,
                                              margin=0.2)
        frame_loss2 = F.cosine_embedding_loss(spk2.flatten(0, 1), spk2_sfl.flatten(0, 1), target=labels) + \
                      F.cosine_embedding_loss(spk1.flatten(0, 1), spk2_sfl.flatten(0, 1), target=-1. * labels,
                                              margin=0.2)
        frame_loss = frame_loss1 + frame_loss2
        s1, s2 = seq_feat.chunk(2, dim=0)  # 不相同  (N, D)
        l = random.choice(range(1, s1.shape[0]))  # [1: T-1]
        s1_sfl = torch.cat((s1[l:, ...], s1[:l, ...]), dim=0).contiguous()
        s2_sfl = torch.cat((s2[l:, ...], s2[:l, ...]), dim=0).contiguous()
        y = torch.ones(s1.shape[0], device=vids.device)
        seq_loss1 = F.cosine_embedding_loss(s1, s1_sfl, target=y) + \
                    F.cosine_embedding_loss(s2, s1_sfl, target=-1. * y, margin=0.2)
        seq_loss2 = F.cosine_embedding_loss(s2, s2_sfl, target=y) + \
                    F.cosine_embedding_loss(s1, s2_sfl, target=-1. * y, margin=0.2)
        seq_loss = seq_loss1 + seq_loss2
        spk_loss = frame_loss + seq_loss
        ## for vsr
        logits, vid_feat, cont_feat = self.vsr(vids, xlens)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        ## for drl
        c1, c2 = cont_feat.chunk(2, dim=0)  # 对应s1, s2   (N, T, D)
        diff_loss = diff_loss(s1.unsqueeze(1), c1) + diff_loss(s2.unsqueeze(1), c2) 
        return vsr_loss + spk_loss + diff_loss

    # 低效！
    def calc_cl_loss(self, vids, tgts, xlens, ylens):
        vids = torch.flatten(vids, 0, 1)
        tgts = torch.flatten(tgts, 0, 1)
        xlens = torch.flatten(xlens, 0, 1)
        ylens = torch.flatten(ylens, 0, 1)
        # vids: (16x2, T, C, H, W)
        logits, cont_feat = self.vsr(vids, xlens)
        spk_feat = self.spk(vids)[1]
        ## for drl
        #spk1, spk2 = spk_feat.chunk(2, dim=0)   # 相同
        #cont1, cont2 = cont_feat.chunk(2, dim=0)  # 不同
        spk1, spk2 = spk_feat[0::2], spk_feat[1::2]
        cont1, cont2 = cont_feat[0::2], cont_feat[1::2]
        feat1 = torch.cat((F.normalize(spk1, dim=-1), F.normalize(cont2, dim=-1)), dim=0)
        feat2 = torch.cat((F.normalize(spk2, dim=-1), F.normalize(cont1, dim=-1)), dim=0)
        scores1 = torch.matmul(F.normalize(spk1, dim=-1), feat2.transpose(0, 1)) / self.tmp
        scores2 = torch.matmul(F.normalize(spk2, dim=-1), feat1.transpose(0, 1)) / self.tmp
        drl_loss = F.cross_entropy(scores1, torch.tensor(list(range(len(spk1))), dtype=torch.long, device=vids.device)) + F.cross_entropy(scores2, torch.tensor(list(range(len(spk2))), dtype=torch.long, device=vids.device))
        ## for vsr
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        vsr_loss = F.ctc_loss(log_probs, tgts, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)
        return vsr_loss + drl_loss * 0.5

    def greedy_decode(self, vids, lens=None):
        return self.avsr.ctc_greedy_decode(vids, lens)

    def beam_decode(self, vid, aud, vid_lens, aud_lens, bos_id, eos_id, max_dec_len=50, pad_id=0):
        return self.avsr.beam_search_decode(vid, aud, vid_lens, aud_lens, bos_id, eos_id, max_dec_len, pad_id)

