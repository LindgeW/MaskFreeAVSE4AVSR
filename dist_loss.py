import torch
import torch.nn as nn
import torch.nn.functional as F


def nuclear_norm_dist(x1, x2):
    nx1 = F.normalize(x1, dim=-1)
    nx2 = F.normalize(x2, dim=-1)
    loss = torch.abs(torch.norm(nx1, 'nuc') - torch.norm(nx2, 'nuc')) / x1.shape[0]
    return loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        mx1 = torch.mean(x1, dim=0)
        mx2 = torch.mean(x2, dim=0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms = scms + self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        #power = torch.pow(x1-x2, 2)
        #summed = torch.sum(power)
        #sqrt = summed ** 0.5
        #return sqrt
        return torch.norm(x1-x2, p=2)   # ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), dim=0)
        ss2 = torch.mean(torch.pow(sx2, k), dim=0)
        return self.matchnorm(ss1, ss2)


def cmd(x1, x2, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i+2))
        # scms+=moment_diff(sx1, sx2, 1)
    return sum(scms) 

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return torch.norm(x1 - x2, p=2)

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    #ss1 = sx1.mean(0)
    #ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm
        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt
        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss


def CORAL2(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)
    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4 * d * d)
    return loss


# Deep CORAL
def CORAL3(source, target):
    DEVICE = source.device
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


# 多核MMD
def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def in_batch_neg_loss(x1, x2, temp=1.):
    """
    Args:
        x1 (torch.Tensor):  (batch_size, dim_size)
        x2 (torch.Tensor):  (batch_size, dim_size)
        temp (float): temperature parameter
    Returns:
        torch.Tensor: 基于批次内的对比学习损失 (in-batch负样本，即在一个batch中，每个样本的负样本是其他样本)
    """
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    sim_mat = torch.matmul(x1, x2.T) / temp
    labels = torch.arange(len(sim_mat), device=x1.device)
    # 双向对比损失
    loss1 = F.cross_entropy(sim_mat, labels)
    loss2 = F.cross_entropy(sim_mat.T, labels)
    return (loss1 + loss2) / 2.


def nt_xent_loss(x1, x2, temp=1.):
    # x1: (B, D)   x2: (B, D)
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    S = torch.matmul(x1, x2.T)
    # 对角线上的元素是正样本对
    positive_pairs = torch.diag(S)
    # 计算负样本对 (非对角线)
    negative_pairs = S - torch.diag(torch.diag(S))
    # 计算logits
    logits = torch.cat([positive_pairs.unsqueeze(1), negative_pairs], dim=1) / temp
    # 计算labels
    labels = torch.zeros(len(logits)).long().to(x1.device)
    # 计算NT-Xent Loss
    loss = F.cross_entropy(logits, labels)
    return loss


def info_nce_loss(x1, x2, temp=1.):
    # x1: (B, D)   x2: (B, D)
    bs = x1.size(0)
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    # 计算点积相似度 (正样本)
    pos_sim = (x1 * x2).sum(dim=1).unsqueeze(1)   # (B, 1)
    # 构造负样本对 (可以有其他更复杂的构造负例的方式)
    neg_x = x2[torch.randperm(bs)]
    neg_sim = torch.matmul(x1, neg_x.T)  # (B, B)
    # 合并正负样本的相似度得分
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temp   # (B, 1+B)
    # 计算labels
    labels = torch.zeros(bs).long().to(x1.device)
    # 计算InfoNCE Loss
    loss = F.cross_entropy(logits, labels)
    return loss


def CMCM_loss(P_A, P_B):
    # P_A: (N, K), 模态A的概率分布
    # P_B: (N, K), 模态B的概率分布
    Pab = torch.matmul(P_A, torch.log(P_B + 1e-10).t())  # (N, N)
    Pba = torch.matmul(P_B, torch.log(P_A + 1e-10).t())  # (N, N)
    #S_code_matrix = -(Pab + Pba)
    S_code_matrix = Pab + Pba
    exp_S = torch.exp(S_code_matrix)
    numerator = torch.diag(exp_S)
    denominator = exp_S.sum(dim=1)
    loss = -torch.log(numerator / denominator + 1e-10).mean()
    return loss


def paired_contrastive_loss(emb_i, emb_j, temp=1.):
    """
    emb_i and emb_j are batches of embeddings, where corresponding indices are pairs z_i, z_j as per SimCLR paper
    """
    bs = emb_i.size(0)
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    r = torch.cat([z_i, z_j], dim=0)
    sim_mat = F.cosine_similarity(r.unsqueeze(1), r.unsqueeze(0), dim=2)  # 2B x 2B x D

    sim_ij = torch.diag(sim_mat, bs)
    sim_ji = torch.diag(sim_mat, -bs)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temp)
    negatives_mask = (1. - torch.eye(bs * 2, bs * 2)).to(sim_mat.device)
    denominator = negatives_mask * torch.exp(sim_mat / temp)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * bs)

    return loss


# 一种基于冗余减少的自监督学习方法(通过最小化互相关矩阵的非对角元素，实现嵌入向量的去相关，从而避免了其他方法中常见的常量解问题)
# 不需要大batch，也不需要非对称机制(如momentum或stop-gradient)
def barlow_twins_loss(z_a, z_b, lambd=5e-3):
    N, D = z_a.size(0), z_a.size(1)
    # normalize repr. along the batch dimension (也可通过一个nn.BatchNorm1d层实现)
    z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
    z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
    # cross-correlation matrix (互相关矩阵)
    # c = torch.matmul(z_a_norm.T, z_b_norm) / (4*N)  # DxD
    c = torch.matmul(z_a_norm.T, z_b_norm) / N  # DxD
    # loss
    c_diff = (c - torch.eye(D, device=c.device)).pow(2)  # DxD
    # multiply off-diagonal elems of c_diff by lambda
    c_diff[~torch.eye(D).bool()] *= lambd
    loss = c_diff.sum()
    return loss / D


# contrastive loss
def ranking_loss(p1, n1, p2, n2, m=1., mode='cos'):
    if mode == 'cos':
        pos_dist = F.cosine_similarity(p1, p2, dim=-1)
        neg_dist1 = F.cosine_similarity(p1, n1, dim=-1)
        neg_dist2 = F.cosine_similarity(p2, n2, dim=-1)
        l1 = torch.clamp(m - pos_dist + neg_dist1, min=0.0).mean()
        l2 = torch.clamp(m - pos_dist + neg_dist2, min=0.0).mean()
        return (l1 + l2) / 2.
    elif mode == 'l2':
        p1, n1 = F.normalize(p1, dim=-1), F.normalize(n1, dim=-1)
        p2, n2 = F.normalize(p2, dim=-1), F.normalize(n2, dim=-1)
        pos_dist = torch.norm(p1 - p2, p=2, dim=-1)
        neg_dist1 = torch.norm(n1 - p1, p=2, dim=-1)
        neg_dist2 = torch.norm(n2 - p2, p=2, dim=-1)
        l1 = torch.clamp(m + pos_dist - neg_dist1, min=0.0).mean()
        l2 = torch.clamp(m + pos_dist - neg_dist2, min=0.0).mean()
        return (l1 + l2) / 2.
    elif mode == 'contrastive':
        p1, n1 = F.normalize(p1, dim=-1), F.normalize(n1, dim=-1)
        p2, n2 = F.normalize(p2, dim=-1), F.normalize(n2, dim=-1)
        pos = torch.norm(p1 - p2, p=2, dim=-1).pow(2)
        neg1 = torch.clamp(m - torch.norm(n1 - p1, p=2, dim=-1), min=0.0).pow(2)
        neg2 = torch.clamp(m - torch.norm(n2 - p2, p=2, dim=-1), min=0.0).pow(2)
        loss = pos.mean() + neg1.mean() + neg2.mean()
        return loss / 2.
    elif mode == 'logexp': 
        #pos_dist = F.cosine_similarity(p1, p2, dim=-1)
        #neg_dist1 = F.cosine_similarity(p1, n1, dim=-1)
        #neg_dist2 = F.cosine_similarity(p2, n2, dim=-1)
        pos_dist = torch.mul(p1, p2).sum(dim=-1)
        neg_dist1 = torch.mul(p1, n1).sum(dim=-1)
        neg_dist2 = torch.mul(p2, n2).sum(dim=-1)
        l1 = -(pos_dist - neg_dist1).sigmoid().log().mean()
        l2 = -(pos_dist - neg_dist2).sigmoid().log().mean()
        return (l1 + l2) / 2.
    elif mode == 'soft_margin':
        pos_dist = torch.norm(p1 - p2, p=2, dim=-1)
        neg_dist1 = torch.norm(p1 - n1, p=2, dim=-1)
        neg_dist2 = torch.norm(p2 - n2, p=2, dim=-1)
        l1 = F.softplus(pos_dist - neg_dist1).mean()
        l2 = F.softplus(pos_dist - neg_dist2).mean()
        return (l1 + l2) / 2.
        # pos_constraint = pos_dist.mean()
        # neg_constraint = 1.0 / (neg_dist1.mean() + 1e-8) + 1.0 / (neg_dist2.mean() + 1e-8)
        # return (l1 + l2 + pos_constraint + neg_constraint) / 4.
        # pos_constraint = pos_dist.mean()
        # neg_constraint = neg_dist1.mean() + neg_dist2.mean()
        # return l1 + l2 + pos_constraint - neg_constraint
    else:
        raise ValueError("Unknown distance type")


def distance_loss(x1, x2, mode='mse', norm=True):
    if mode == 'cos':
        return (1. - F.cosine_similarity(x1, x2, dim=-1)).mean()
        # return (1. - F.cosine_similarity(x1, x2, dim=-1)).pow(2).mean()
    elif mode == 'l1':
        if norm:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        # return F.l1_loss(x1, x2)
        return F.smooth_l1_loss(x1, x2)
    elif mode == 'l2':
        if norm:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        return torch.norm(x1 - x2, p=2, dim=-1).mean()
        # return torch.norm(x1 - x2, p=2, dim=-1).pow(2).mean()
    elif mode == 'mse':
        if norm:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        return F.mse_loss(x1, x2)  # 相当于L2范数的平方，并除以BD
    else:
        raise ValueError("Unimplemented distance type")


def diff_loss(x1, x2, mode='l2'):  # (B, D1)  (B, D2)
    #if x1.ndim == 3 and x2.ndim == 3:
    #    x1 = torch.flatten(x1, 0, 1)
    #    x2 = torch.flatten(x2, 0, 1)
    if mode == 'l2':
        nx1 = F.normalize(x1, dim=-1)   # 先L2归一化约束在[-1, 1]之间
        nx2 = F.normalize(x2, dim=-1)
        return torch.mean(torch.clamp(1. - torch.norm(nx1 - nx2, p=2, dim=-1), min=0.0).pow(2))
    elif mode == 'euclid':
        # return torch.exp(-0.5 * torch.norm(x1 - x2, p=2, dim=-1)).mean()
        # return torch.mean(1. / (torch.norm(x1 - x2, p=2, dim=-1) + 1e-8))
        return torch.mean(1. / (1. + torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))))   # (0, 1]
    elif mode == 'explog':
        return F.softplus(-torch.norm(x1 - x2, p=2, dim=-1)).mean()
    elif mode == 'fnorm':
        #nx1 = F.normalize(x1 - torch.mean(x1, 0), dim=-1)
        #nx2 = F.normalize(x2 - torch.mean(x2, 0), dim=-1)
        #return torch.mean(torch.matmul(nx1.transpose(-1, -2), nx2).pow(2))  # D1 x D2
        # return torch.mean(torch.matmul(x1.transpose(-1, -2), x2).pow(2))  # D1 x D2
        return DecorrLoss(x1, x2)   # 相关性损失 (去相关)
    elif mode == 'cos':
        return torch.mean(torch.clamp(F.cosine_similarity(x1, x2, dim=-1), min=0.0))  
        #return torch.mean(torch.clamp(F.cosine_similarity(x1-torch.mean(x1, 0), x2-torch.mean(x2, 0), dim=-1), min=0.0))  # 皮尔逊相关性
        #return F.cosine_embedding_loss(x1, x2, torch.tensor([[-1]]).to(x1.device))  # (B, D)
    else:
        raise ValueError("Unknown distance type")


def hsic(x, y, sigma_xy=1.0):
    """
    计算HSIC值
    参数:
    x (torch.Tensor): (n_samples, n_features_x)
    y (torch.Tensor): (n_samples, n_features_y)
    sigma (float): 高斯核的带宽参数
    返回:
    hsic_value (float): HSIC值
    """
    def linear_kernel(X):
        """
        计算线性核矩阵。
        """
        X_centered = X - X.mean(dim=0)
        K = X_centered @ X_centered.T
        return K / X.size(0)

    def kernel_matrix(x, sigma=1.0):
        dim = len(x.size())
        x1 = torch.unsqueeze(x, 0)
        x2 = torch.unsqueeze(x, 1)
        axis = tuple(range(2, dim + 1))
        if dim > 1:
            return torch.exp(-0.5 * torch.sum(torch.pow(x1 - x2, 2), axis=axis) / sigma ** 2)
        else:
            return torch.exp(-0.5 * torch.pow(x1 - x2, 2) / sigma ** 2)

    def rbf_kernel(X, gamma=1.0):
        """RBF核函数"""
        # gamma = 1. / X.size(-1)
        X_norm = torch.sum(X ** 2, dim=1)
        K = torch.exp(-gamma * (X_norm[:, None] + X_norm[None, :] - 2 * X @ X.T))
        return K

    def gaussian_kernel(X, sigma=None):   # RBF核矩阵
        """
        计算高斯核矩阵
        参数:
        X (torch.Tensor): 特征矩阵， (n_samples, n_features)
        sigma (float): 高斯核的带宽参数
        返回:
        kernel_matrix (torch.Tensor): 高斯核矩阵， (n_samples, n_samples)
        """
        sq_dists = torch.cdist(X, X) ** 2
        if sigma is None:
            # sigma = 1.0
            sigma = X.size(-1)   # feat dim
            # sigma = X.size(-1)**0.5  # sqrt of feat dim
            # sigma = torch.median(sq_dists.flatten())     # 中值启发式
            # sigma = torch.median(sq_dists[sq_dists > 0])  # 中值启发式
        gamma = 1. / (2 * sigma)
        # gamma = 1. / (2 * sigma**2)   # for sqrt of feat dim
        K = torch.exp(-sq_dists * gamma)
        return K

    n = x.size(0)
    # 计算核矩阵 K 和 L
    kx = gaussian_kernel(x, sigma_xy)  # sigma=1更好
    ky = gaussian_kernel(y, sigma_xy)
    # 中心化核矩阵
    h = torch.eye(n, device=x.device) - torch.ones((n, n), device=x.device) / n
    kxc = h @ kx @ h
    kyc = h @ ky @ h
    # 计算HSIC
    hsic_value = torch.trace(kxc @ kyc) / (n - 1)**2   # work
    # hsic_value = torch.trace(kxc.T @ kyc) / (n - 1)**2
    # hsic_value = torch.sum(kxc * kyc) / (n - 1)**2   # faster
    # hsic_value = torch.trace(kx @ h @ ky @ h) / (n - 1)**2  # 也看到有这么算的
    return hsic_value


# (B, D)
def decorr_loss(h1, h2):
    # 计算协方差矩阵
    h1_centered = h1 - h1.mean(dim=0, keepdims=True)
    h2_centered = h2 - h2.mean(dim=0, keepdims=True)
    # 交叉协方差矩阵
    cov_matrix = torch.matmul(h1_centered.T, h2_centered) / (h1.size(0) - 1)
    # 去相关损失为交叉协方差矩阵的Frobenius范数
    loss = torch.norm(cov_matrix, p='fro')
    return loss

# (B, T, D)
def DecorrLoss(h1, h2):
    if h1.ndim == 3:
        # 将三维的数据展平为二维
        B, T, C = h1.size()
        # 计算每个时间步的均值
        h1_mean = h1.mean(dim=1, keepdim=True)
        h2_mean = h2.mean(dim=1, keepdim=True)
        # 去中心化
        h1_centered = h1 - h1_mean
        h2_centered = h2 - h2_mean
        # 将每个时间步的数据拼接在一起
        h1_centered = h1_centered.reshape(-1, C)  # (B*T, D)
        h2_centered = h2_centered.reshape(-1, C)  # (B*T, D)
    else:
        # 计算协方差矩阵
        h1_centered = h1 - h1.mean(dim=0, keepdims=True)
        h2_centered = h2 - h2.mean(dim=0, keepdims=True)
    # 计算交叉协方差矩阵
    cov_matrix = torch.matmul(h1_centered.T, h2_centered) / (h1_centered.size(0) - 1)
    # 去相关损失为交叉协方差矩阵的 Frobenius 范数
    loss = torch.norm(cov_matrix, p='fro')  # 平方和开方
    return loss


def orth_loss(x1, x2):
    batch_size = x1.size(0)
    x1 = x1.reshape(batch_size, -1)
    x2 = x2.reshape(batch_size, -1)

    x1_mean = torch.mean(x1, dim=0, keepdims=True)
    x2_mean = torch.mean(x2, dim=0, keepdims=True)
    x1 = x1 - x1_mean
    x2 = x2 - x2_mean

    x1_l2_norm = torch.norm(x1, p=2, dim=1, keepdim=True).detach()
    x1_l2 = x1.div(x1_l2_norm.expand_as(x1) + 1e-8)

    x2_l2_norm = torch.norm(x2, p=2, dim=1, keepdim=True).detach()
    x2_l2 = x2.div(x2_l2_norm.expand_as(x2) + 1e-8)

    diff_loss = torch.mean((x1_l2.t().mm(x2_l2)).pow(2))
    return diff_loss


class JSDLoss(nn.Module):
    def __init__(self, hid_size1, hid_size2, norm=False):
        super(JSDLoss, self).__init__()
        self.norm = norm
        hid_size = hid_size1 + hid_size2
        self.net = nn.Sequential(nn.Linear(hid_size, hid_size//2),   # 可以换成Conv(k=1, s=1)
                                 nn.ReLU(),
                                 #nn.Linear(hid_size//2, hid_size//2))
                                 #nn.ReLU(),
                                 nn.Linear(hid_size//2, 1))
        #self.fc = nn.Linear(hid_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        bs = x.size(0)
        tiled_x = torch.cat((x, x), dim=0)
        shuffled_y = torch.cat((y[1:], y[0].unsqueeze(0)), dim=0)
        concat_y = torch.cat((y, shuffled_y), dim=0)
        inputs = torch.cat((tiled_x, concat_y), dim=-1)
        logits = self.net(inputs)
        #logits = self.fc(F.relu(inputs))
        if self.norm:
            logits = F.normalize(logits, p=2, dim=-1)
        pred_xy = logits[:bs]
        pred_x_y = logits[bs:]
        mi_loss = (-F.softplus(-pred_xy)).mean() - F.softplus(pred_x_y).mean()  # max jsd
        return -mi_loss


def jsd_loss(T: torch.Tensor, T_prime: torch.Tensor):
    """Estimator of the Jensen Shannon Divergence see paper equation (2)

      Args:
        T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
        T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

      Returns:
        float: DJS estimation value
    """
    joint_expectation = (-F.softplus(-T)).mean()   # pxy
    marginal_expectation = F.softplus(T_prime).mean()  # pxpy
    mutual_info = joint_expectation - marginal_expectation
    return -mutual_info


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):   # 用于最小化x和y之间互信息的训练loss
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return positive.sum(dim=-1) - negative.sum(dim=-1)

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):   # club loss用于更新club模块参数 (一般需要单独用optimizer更新CLUB若干轮)
        return -self.loglikeli(x_samples, y_samples)
