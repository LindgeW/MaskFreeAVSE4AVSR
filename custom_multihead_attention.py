import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # 添加batch维度
        self.register_buffer('pe', pe)   # 不可训练

    def forward(self, x):
        """
        x: [B, L, D]
        """
        return self.dropout(x + self.pe[:, :x.size(1)])


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, max_pos_embs, emb_dim, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embs = nn.Embedding(max_pos_embs, emb_dim)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        pos_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)   # 创建位置索引
        pos_embs = self.pos_embs(pos_ids)    # 获取位置编码
        return self.dropout(x + pos_embs)



"""基于Conv1d的多头注意力机制"""
class ConvMultiheadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        self.q_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):   # (B, L, D)
        B, L, D = q.shape

        q = self.q_proj(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_proj(k.transpose(1, 2)).transpose(1, 2)
        v = self.v_proj(v.transpose(1, 2)).transpose(1, 2)

        q = q.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.out_proj(out.transpose(1, 2)).transpose(1, 2)
        return out   # (B, L, D)



class ConvMultiheadAttention2(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size=1, dropout=0.1):
        super(ConvMultiheadAttention2, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        	query: [B, L, D]
            key: [B, L, D]
            value: [B, L, D]
            key_padding_mask: [B, L]
            attn_mask: [L, L]
        """
        q = self.q_proj(query.transpose(1, 2))  # [B, E, L]
        k = self.k_proj(key.transpose(1, 2))    # [B, E, L]
        v = self.v_proj(value.transpose(1, 2))  # [B, E, L]
        
        B, D, L = q.size()
        
        q = q.reshape(B, self.num_heads, self.head_dim, L)  # [B, H, D, L]
        k = k.reshape(B, self.num_heads, self.head_dim, L)  # [B, H, D, L]
        v = v.reshape(B, self.num_heads, self.head_dim, L)  # [B, H, D, L]
        
        # [B, H, L, L]
        attn_weights = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        
        # 应用注意力掩码
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0).unsqueeze(1)
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.einsum('bhij,bhdj->bhdi', attn_weights, v)  # [B, H, D, L]
        
        output = output.reshape(B, -1, L)  # [B, E, L]
        
        output = self.out_proj(output)  # [B, E, L]
        
        output = output.transpose(1, 2)  # [B, L, E]
        
        return output



class ConvMLP(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, dim * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 4, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):   # (B, L, D)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        return x



class ConvAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, batch_first=False, use_residual=True, is_cross_attn=False, has_mlp=True, norm_first=False):
        super().__init__()
        self.use_residual = use_residual
        self.norm_first = norm_first
        self.has_mlp = has_mlp
        #self.attn = ConvMultiheadAttention(dim, heads, dropout)
        self.attn = ConvMultiheadAttention2(dim, heads, 1, dropout)
        self.mlp = ConvMLP(dim, dropout) if has_mlp else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if is_cross_attn and norm_first else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
        self.drop_path = DropPath(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q, kv=None):
        if self.norm_first:
            norm_q = self.norm1(q)
            if kv is not None:
                norm_kv = self.norm2(kv)
                out = self.attn(norm_q, norm_kv, norm_kv)
            else:
                out = self.attn(norm_q, norm_q, norm_q)
            x = q + self.drop_path(out) if self.use_residual else out
            if self.has_mlp:
                x = x + self.drop_path(self.mlp(self.norm3(x)))
            return x
        else:
            if kv is not None:
                out = self.attn(q, kv, kv)
            else:
                out = self.attn(q, q, q)
            x = self.norm1(q + self.drop_path(out)) if self.use_residual else self.norm1(out)
            if self.has_mlp:
                x = self.norm3(x + self.drop_path(self.mlp(x)))
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




