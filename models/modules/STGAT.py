# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from torch.nn import Linear, BatchNorm1d
from torch_geometric.utils import dense_to_sparse


class TemporalAttention(torch.nn.Module):
    """
        model imput: (batch_size, num_of_nodes, channels, time_step)
    """

    def __init__(self, num_of_nodes, channels, time_step):
        super().__init__()
        self.W_1 = nn.Parameter(torch.zeros(size=(num_of_nodes,)))
        nn.init.uniform_(self.W_1.data)
        self.W_2 = nn.Parameter(torch.zeros(size=(channels, time_step)))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, time_step, time_step)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.V = nn.Parameter(torch.zeros(size=(time_step, time_step)))
        nn.init.xavier_uniform_(self.V.data, gain=1.414)

        self.bn = nn.BatchNorm1d(time_step)

    def forward(self, x):
        product = torch.matmul(torch.matmul(
            x.permute(0, 3, 2, 1), self.W_1), self.W_2)

        E = torch.tanh(torch.matmul(product, self.V)+self.b)
        E_normalized = self.bn(E)
        return E_normalized


class SpatialAttention(torch.nn.Module):
    """
        model imput: (batch_size, num_of_nodes, channels, time_step)
    """

    def __init__(self, num_of_nodes, channels, time_step):
        super().__init__()
        self.W_1 = nn.Parameter(torch.zeros(size=(time_step,)))
        nn.init.uniform_(self.W_1.data)
        self.W_2 = nn.Parameter(torch.zeros(size=(channels, num_of_nodes)))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(
            size=(1, num_of_nodes, num_of_nodes)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.V = nn.Parameter(torch.zeros(size=(num_of_nodes, num_of_nodes)))
        nn.init.xavier_uniform_(self.V.data, gain=1.414)
        self.bn = BatchNorm1d(num_of_nodes)

    def forward(self, x):
        product = torch.matmul(torch.matmul(x, self.W_1), self.W_2)

        S = torch.tanh(torch.matmul(product, self.V)+self.b)

        S_normalized = self.bn(S)
        return S_normalized


class SOGAT(torch.nn.Module):
    """Self-organized Graph Construction Module
    Args:
        in_features: size of each input sample
        bn_features: size of bottleneck layer
        out_features: size of each output sample
        topk: size of top k-largest connections of each channel
    """

    def __init__(self, time_step: int, bn_features: int, out_features: int, conv_channels: int, topk: int, spatial: bool, temporal: bool):
        super().__init__()

        self.num_of_vertices = 62
        self.time_step = time_step
        self.bn_features = bn_features
        self.out_features = out_features
        self.conv_channels = conv_channels
        self.topk = topk
        self.spatial = spatial
        self.temporal = temporal

        self.bnlin = Linear(time_step*conv_channels, bn_features)

        self.gconv1 = GATConv(time_step*conv_channels,
                              out_features, heads=4, dropout=0.3)
        self.gconv2 = GATConv(out_features*4,
                              out_features, heads=4, concat=False, dropout=0.3)
        # self.gconv1 = DenseGCNConv(
        #     time_step*conv_channels, out_features)
        # self.gconv2 = DenseGCNConv(out_features, out_features)
        self.s_attr = None
        self.t_attr = None
        if spatial:
            self.s_attr = SpatialAttention(
                self.num_of_vertices, conv_channels, time_step)
        if temporal:
            self.t_attr = TemporalAttention(
                self.num_of_vertices, conv_channels, time_step)

    def forward(self, x, edge_index):

        x = x.reshape(-1, self.num_of_vertices,
                      self.conv_channels, self.time_step)

        if self.spatial:
            adj = self.s_attr(x)
            amask = torch.zeros(adj.size(0), self.num_of_vertices,
                                self.num_of_vertices).cuda()
            amask.fill_(0.0)
            s, t = adj.topk(self.topk, 2)
            amask.scatter_(2, t, s.fill_(1))
            adj = adj*amask
            edge_index, _ = dense_to_sparse(adj)

        if self.temporal:
            temporal_attr = self.t_attr(x)
            torch.matmul(x.reshape(-1, self.conv_channels*self.num_of_vertices, self.time_step),
                         temporal_attr).reshape(-1, self.num_of_vertices, self.conv_channels, self.time_step)

        x = x.reshape(-1, self.conv_channels*self.time_step)

        x = F.relu(self.gconv1(x, edge_index))
        x = F.relu(self.gconv2(x, edge_index))
        return x
