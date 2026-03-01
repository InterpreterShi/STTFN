import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
from math import sqrt
from utils.weight_load import WeightProcess

'''
空间权重是由不同的空间节点拓扑关系组成的，是先验的信息
* 一种是收集到了一定的先验信息，语义权重关系由先验关系通过神经网络学习
* 另一种是无法获取先验信息，使用自适应的权重关系，参数化学习
* so,如何结合，并且具有动态空间权重关系？
* 建模：全局尺度因子lambda, 局部尺度因子mu
'''


class convt(nn.Module):
    def __init__(self):
        super(convt, self).__init__()

    def forward(self, x, w):
        x = torch.einsum('bne, ek->bnk', (x, w))
        return x.contiguous()


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, C, dims):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, C))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, C))
        else:
            raise NotImplementedError('not implemented for dimension')
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    """Graph convolution network."""

    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.c_in = c_in
        c_in = (order * support_len + 1) * self.c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for c in support:
            x1 = self.nconv(x, c.to(x.device), c.dim())
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, c.to(x1.device), c.dim())
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout)
        return h


def dy_mask_graph(adj, k):
    M = []
    for i in range(adj.size(0)):
        adp = adj[i]
        mask = torch.zeros(adj.size(1), adj.size(2)).to(adj.device)
        mask = mask.fill_(float("0"))
        s1, t1 = (adp + torch.rand_like(adp) * 0.01).topk(k, dim=1)
        mask = mask.scatter_(1, t1, s1.fill_(1))
        M.append(mask)
    mask = torch.stack(M, dim=0)
    adj = adj * mask
    return adj


def cat(x1, x2):
    M = []
    for i in range(x1.size(0)):
        x = x1[i]
        new_x = torch.cat([x, x2], dim=1)
        M.append(new_x)
    result = torch.stack(M, dim=0)
    return result


# 新引入模块
class DFDB(nn.Module):
    def __init__(self, in_len, num_nodes, embed_dim, identity_emb=10, hidden_emb=30, subgraph=20, dropout=0.3):
        super(DFDB, self).__init__()
        self.in_len = in_len
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.identity_emb = identity_emb
        self.hidden_emb = hidden_emb
        self.fft_len = round(in_len // 2) + 1
        self.Ex = nn.Parameter(torch.randn(self.fft_len, self.embed_dim), requires_grad=True)
        self.nodes = nn.Parameter(torch.randn(num_nodes, self.identity_emb), requires_grad=True)
        self.Wd = nn.Parameter(torch.randn(num_nodes, self.embed_dim + self.identity_emb, self.hidden_emb),
                               requires_grad=True)
        self.Wxabs = nn.Parameter(torch.randn(self.hidden_emb, self.hidden_emb), requires_grad=True)
        self.layersnorm = torch.nn.LayerNorm(normalized_shape=[num_nodes, self.hidden_emb], eps=1e-08,
                                             elementwise_affine=False)
        self.drop = nn.Dropout(p=dropout)
        self.subgraph_size = subgraph
        self.convt = convt()

    def forward(self, x):  # x:[B, T, N, C]
        """
        计算频域关系 frequency_supports（基于频域信息）
        """
        x_ = x[:, :, :, 0]  # [B, T, N]
        x_ = x_.permute(0, 2, 1).contiguous()  # [B, N, T]
        x_ = torch.fft.rfft(x_, dim=-1)
        x_ = torch.abs(x_)
        x_ = torch.nn.functional.normalize(x_, p=2.0, dim=1, eps=1e-12, out=None)
        x_ = torch.nn.functional.normalize(x_, p=2.0, dim=2, eps=1e-12, out=None)
        x_ = torch.matmul(x_, self.Ex)  # [B, N, emb]
        x_k = cat(x_, self.nodes)  # [B, N, emb+emb_iden]
        x1 = torch.bmm(x_k.permute(1, 0, 2), self.Wd).permute(1, 0, 2)  # [B, N, hidden_emb]
        x1 = torch.relu(x1)
        x2 = self.layersnorm(x1)
        x2 = self.drop(x2)
        adp = self.convt(x2, self.Wxabs)  # [B, N, hidden_emb]
        adj = torch.bmm(adp, x1.permute(0, 2, 1))  # [B, N, N]
        adp = torch.relu(adj)
        adp = dy_mask_graph(adp, self.subgraph_size)
        frequency_supports = F.softmax(adp, dim=2)

        return frequency_supports


# 自适应动态空间权重
class DASW(nn.Module):
    def __init__(self, in_len, num_nodes, embed_dim, g_lambda=0.5, l_mu=0.5):
        super(DASW, self).__init__()
        self.in_len = in_len
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.g_lambda = nn.Parameter(torch.tensor(g_lambda, dtype=torch.float32, requires_grad=True))
        self.l_mu = nn.Parameter(torch.tensor(l_mu, dtype=torch.float32, requires_grad=True))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, self.embed_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(self.embed_dim, num_nodes), requires_grad=True)

    def scaled_Laplacian(self, s_w):
        """
        计算的是 L_hat = 2/lambda_max - I
        lambda_max 是拉普拉斯矩阵的最大特征值
        """
        D = torch.diag(torch.sum(s_w, dim=1))
        L = D - s_w
        L = torch.nan_to_num(L)
        lambda_max = max([torch.nan_to_num(torch.real(i)) for i in torch.linalg.eigvals(L)])
        return torch.nan_to_num((2 * L) / lambda_max - torch.eye(s_w.shape[0]).to(L.device))

    def forward(self, s_w):  # s_w: [N, N]
        """
        计算动态自适应的空间权重:
        1. 计算局部关系 local_supports（基于可学习嵌入）
        2. 计算全局关系 global_supports（基于特征值分解）
        3. 使用可训练的 g_lambda、l_mu进行加权
        """
        # 计算局部邻接矩阵
        local_supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # [N, N]
        global_supports = self.scaled_Laplacian(s_w)  # [N, N]

        total = self.g_lambda + self.l_mu

        return self.g_lambda / total.sum() * global_supports + self.l_mu / total.sum() * local_supports


class SRGCN(nn.Module):
    def __init__(self, in_len, num_nodes, embed_dim, in_dim, out_dim, spatial_attention=False):
        super(SRGCN, self).__init__()
        self.in_len = in_len
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_attention = spatial_attention

        self.dasw = DASW(in_len, num_nodes, embed_dim)
        self.dfdb = DFDB(in_len, num_nodes, embed_dim)

        # self.gconv = gcn(in_dim, out_dim, dropout=0.3)
        self.gconv1 = gcn(in_dim, 64, dropout=0.3)
        self.gconv2 = gcn(64, out_dim, dropout=0.3)

    def forward(self, s_w, x):
        B, T, N, C = x.shape

        supports_1 = self.dasw(s_w)  # [N, N]
        supports_2 = self.dfdb(x)  # [B, N, N]
        supports = [supports_1] + [supports_2]

        l_n = torch.ones((self.in_len, self.num_nodes)).to(x.device)
        w_x_1 = torch.einsum('btnc, nn->btnc', x, supports_1)
        w_x_2 = torch.einsum('btnc, bnn->btnc', x, supports_2)
        x_t_hat = torch.concat((l_n.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, 1), x, w_x_1, w_x_2),
                               dim=-1)  # [B,T,N,3C+1]

        x_0 = x.transpose(1, 3).contiguous()  # [B, C, N, T]
        x1 = self.gconv1(x_0, supports)
        x2 = self.gconv2(F.relu(x1), supports)
        out = x2.transpose(1, 3).contiguous()
        # out_0 = self.gconv(x_0, supports) # [B, F, N, T]
        # out = out_0.transpose(1, 3).contiguous() # [B, T, N, F]

        supports_0 = 0.5 * supports_1 + 0.5 * supports_2.mean(dim=0)  # [N, N]

        if self.spatial_attention:
            return F.relu(out), supports_0, x_t_hat
        return F.relu(out), None, x_t_hat


if __name__ == '__main__':
    s_w = WeightProcess(root_path='../dataset', num_nodes=307, dataset='pems04').s_w
    print(s_w.min())
    # device = torch.device('cuda')
    # s_w = torch.rand(307,307).to(device)
    # x = torch.rand(64,12,307,3).to(device)
    # # model = DASW(in_len=12, num_nodes=307, embed_dim=10, g_lambda=0.5, l_mu=0.5)
    # model1 = SRGCN(in_len=12, num_nodes=307, embed_dim=10, g_lambda=0.5, l_mu=0.5,
    #               in_dim=3, out_dim=512, cheb_k=2, spatial_attention=True).to(device)
    # # supports= model(s_w)
    # out, s_attn, x_t_hat = model1(s_w, x)
    # # print(supports.shape)
    # print(out.shape)
    # print(s_attn.shape)
    # print(x_t_hat.shape)