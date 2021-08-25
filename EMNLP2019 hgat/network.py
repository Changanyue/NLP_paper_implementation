# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions import uniform
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    GCN层

    """
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super(GraphConvolutionLayer, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.allow_bias = bias

        # 初始化网络权重
        std = 1.0 / math.sqrt(out_features)
        sampler = uniform.Uniform(-std, std)
        self.weight = nn.Parameter(sampler.sample([in_features, out_features]), requires_grad=True).to(device)
        if bias:
            self.bias = nn.Parameter(sampler.sample([out_features]), requires_grad=True).to(device)
        else:
            self.bias = None

    def forward(self, h, adj_norm):
        h_w = torch.matmul(h, self.weight)
        out = torch.matmul(adj_norm, h_w)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f'GraphConvolutionLayer(in_features={self.in_features},out_features={self.out_features},bias={self.allow_bias},device="{self.device}")'


class GraphConvolutionNet(nn.Module):
    """
    GCN网络，本次实验没有用到

    """
    def __init__(self, n_feature, n_hidden, n_class, n_layer=0, bias=False, device='cpu'):
        super(GraphConvolutionNet, self).__init__()
        self.device = device

        # 建立网络 n_hidden为隐层节点数，n_layer为隐层数
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolutionLayer(n_feature, n_hidden, bias))
        for _ in range(n_layer):
            self.layers.append(GraphConvolutionLayer(n_hidden, n_hidden, bias))
        self.layers.append(GraphConvolutionLayer(n_hidden, n_class, bias))
        self.layers.to(device)

    def forward(self, h, adj):
        # 计算归一化邻接矩阵
        n_samples = adj.shape[0]
        adj_norm = adj + torch.eye(n_samples).to(self.device)
        m = torch.diag(torch.sqrt(torch.tensor(1).to(self.device) / torch.sum(adj_norm, dim=1)))
        adj_norm = torch.matmul(torch.matmul(m, adj_norm), m)

        # 逐层计算输出
        hs = []
        for layer in self.layers[0:-1]:
            h = layer(h, adj_norm)
            h = torch.relu(h)

        # 输出层使用sigmoid激活函数
        h = self.layers[-1](h, adj_norm)
        h = torch.sigmoid(h)
        return torch.softmax(h, dim=1)


class Attention(nn.Module):
    """
    注意力模型

    """
    def __init__(self, in_features, hidden_features, type_idx, device='cpu'):
        super(Attention, self).__init__()
        self.type_idx = type_idx
        self.hidden_layer = nn.Linear(in_features, hidden_features).to(device) # 隐层为普通的神经网络层
        self.device = device
        # 初始化注意力权重向量
        sampler = uniform.Uniform(-1.0, 1.0)
        self.attention = nn.Parameter(sampler.sample([2 * hidden_features, 1]), requires_grad=True).to(device)

    def forward(self, h_all):
        # 计算注意力矩阵
        h = self.hidden_layer(h_all).transpose(0, 1) # 由于h_all是3D张量，此次矩阵乘法在前两个维度broadcast
        depth = h.shape[0] # 节点类型数，本次实验depth=3
        h = torch.cat([h, torch.stack([h[self.type_idx]] * depth, dim=0)], dim=2)

        weights = torch.matmul(h, self.attention.to(self.device)).transpose(0, 1)
        weights = nn.LeakyReLU()(weights)
        weights = torch.log_softmax(weights, dim=1)

        # 根据论文公式7计算最终特征
        h_out = torch.matmul(weights.transpose(1, 2), h_all).squeeze(1)
        return weights, h_out


class HGraphConvolutionNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_class, n_layer, bias, device='cpu', dropout=0.95, allow_attention=True):
        super(HGraphConvolutionNet, self).__init__()
        # Network parameters.
        self.device = device
        self.layers = nn.ModuleList() # 最终会有 n_layer * n_module 个GCN层
        self.allow_attention = allow_attention
        self.attentions = nn.ModuleList() # 最终会有 n_layer * n_module 个Attention模块
        self.n_class = n_class
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.n_modules = len(n_features)
        self.dropout = dropout

        for idx_i in range(self.n_modules):
            # 初始化子模块
            module = nn.ModuleList()
            attention = nn.ModuleList()

            # 新建输入层
            input_layer = GraphConvolutionLayer(n_features[idx_i], n_hidden[0], bias)
            module.append(input_layer)

            attention_input = Attention(in_features=n_hidden[0], hidden_features=50, type_idx=idx_i, device=device)
            attention.append(attention_input)

            # 新建隐层
            for idx_j in range(len(n_hidden) - 1):
                hidden_layer = GraphConvolutionLayer(in_features=n_hidden[idx_j], out_features=n_hidden[idx_j + 1],
                                                     bias=bias)
                module.append(hidden_layer)
                attention_hidden = Attention(in_features=n_hidden[idx_j + 1], hidden_features=50, type_idx=idx_i,
                                             device=device)
                attention.append(attention_hidden)
            # 新建输出层
            output_layer = GraphConvolutionLayer(in_features=n_hidden[-1], out_features=n_class, bias=bias)
            module.append(output_layer)
            attention_output = Attention(in_features=n_class, hidden_features=50, type_idx=idx_i, device=device)
            attention.append(attention_output)

            # 保存子模块
            self.layers.append(module)
            self.attentions.append(attention)

        # CPU or CUDA.
        self.layers.to(device)
        self.attentions.to(device)

    def forward(self, hs, adj, splits, normalize=True):
        h_out = None
        n_samples = adj.shape[0]
        # 如果没有归一化，则使用论文公式1进行归一化
        if normalize:
            adj_norm = adj + torch.eye(n_samples).to(self.device)
            m = torch.diag(torch.sqrt(torch.tensor(1).to(self.device) / torch.sum(adj_norm, dim=1)))
            adj_norm = torch.matmul(torch.matmul(m, adj_norm), m)
        else:
            adj_norm = adj
        if self.allow_attention:
            # 计算带注意力的网络，参考论文公式7
            for idx_layer in range(self.n_layer):
                h_out_list = []
                for idx_module_1 in range(self.n_modules):
                    h_layers = [] # 各类型间的特征
                    for idx_module_2 in range(self.n_modules):
                        A_r = adj_norm[splits[idx_module_1], :][:, splits[idx_module_2]] # 获取邻接矩阵的一部分
                        layer = self.layers[idx_module_2][idx_layer] # 计算隐层特征
                        if h_out is None:
                            h_r = hs[idx_module_2]
                        else:
                            h_r = h_out[splits[idx_module_2], :]
                        h_r_next = layer(h_r, A_r)
                        h_layers.append(h_r_next)
                    _, h_out_part = self.attentions[idx_module_1][idx_layer](torch.stack(h_layers, dim=1))
                    h_out_list.append(h_out_part)
                h_out = torch.cat(h_out_list)

                if idx_layer != self.n_layer - 1:
                    h_out = torch.relu(h_out)
                    h_out = F.dropout(h_out, self.dropout, self.training)
        else:
            # 计算不带注意力的网络，参考论文公式2
            h_next = None
            for idx_layer in range(self.n_layer):
                for idx_module in range(self.n_modules):
                    A_r = adj_norm[:, splits[idx_module]]
                    layer = self.layers[idx_module][idx_layer]
                    if h_out is None:
                        h_r = hs[idx_module]
                    else:
                        h_r = h_out[splits[idx_module], :]
                    h_r_next = layer(h_r, A_r)
                    if h_next is None:
                        h_next = h_r_next
                    else:
                        h_next = h_next + h_r_next
                h_out = h_next
                if idx_layer != self.n_layer - 1:
                    h_out = torch.relu(h_out)
                    h_out = F.dropout(h_out, self.dropout, self.training)
                h_next = None
        # 输出层使用sigmoid激活
        h_out = torch.sigmoid(h_out)
        return torch.softmax(h_out, dim=1)
