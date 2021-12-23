# -*- coding: utf-8 -*-
"""

"""
import math
from numpy import product
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, DenseSAGEConv, dense_diff_pool, DenseGCNConv, GATConv, GATv2Conv
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch.nn import Linear, Dropout, PReLU, Conv2d, MaxPool2d, AvgPool2d, Parameter, BatchNorm2d, Dropout2d, Sequential, ReLU6, ReLU, TransformerEncoderLayer, TransformerEncoder, BatchNorm1d
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

from ..modules.conv_bn_relu import ConvBNReLU
from ..modules.STGAT import STGAT


class in_encoder_sogat2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gat1_params = config['gat1']
        self.gat2_params = config['gat2']
        self.drop_rate = config['drop_rate']
        self.heads1 = config['heads1']
        self.heads2 = config['heads2']
        self.channels = config['channels']
        self.encoders_num = config['encoders_num']

        encoder_layer1 = TransformerEncoderLayer(
            d_model=261, nhead=1, batch_first=True, activation="gelu", dropout=0.3)
        self.encoder1 = TransformerEncoder(
            encoder_layer1, num_layers=self.encoders_num, norm=BatchNorm1d(64))

        self.conv1 = nn.Sequential(
            ConvBNReLU(1, 16, (1, 1)),
            ConvBNReLU(16, 32, (3, 3)),
            ConvBNReLU(32, 64, (3, 3))
        )
        self.gcn1_1 = STGAT(time_step=261, bn_features=128, out_features=self.gat2_params,
                            conv_channels=64, topk=10, spatial=True, temporal=False)

        self.gcn2_1 = STGAT(time_step=261, bn_features=128, out_features=self.gat2_params,
                            conv_channels=64, topk=10, spatial=False, temporal=True)

        # self.gcn3_1 = SOGAT(time_step=261, bn_features=128, out_features=self.gat2_params,
        #                     conv_channels=64, topk=10, spatial=True, temporal=True)

        self.pool = AvgPool2d((5, 5))
        self.bn1 = BatchNorm1d(565)
        self.linend = Linear(self.channels*565, 3)

    def forward(self, x, edge_index, batch):
        x, mask = to_dense_batch(x, batch)
        # (Batch*channels, 1, Freq_bands, Features)

        x = x.reshape(-1, 1, 5, 265)
        x0 = self.conv1(x)
        x0 = F.dropout(x0, p=self.drop_rate, training=self.training)
        x0 = x0.reshape(-1, 64, 261)
        x0 = self.encoder1(x0)
        x0 = F.dropout(x0, p=self.drop_rate, training=self.training)

        x1 = self.gcn1_1(x0, edge_index).reshape(-1, self.gat2_params)

        x2 = self.gcn2_1(x0, edge_index).reshape(-1, self.gat2_params)

        x3 = self.pool(x).reshape(-1, 53)
        # x3 = self.gcn3_1(x).reshape(-1, self.gat2_params)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn1(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = x.reshape(-1, self.channels*565)
        x = self.linend(x)
        pred = F.log_softmax(x, 1)

        return x, pred
