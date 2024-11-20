# model.py
# Author: zyw
# Date: 2024-10-23
# Description: Defines the Hybrid GNN model combining GAT and Transformer layers, integrating CEW values into the attention mechanism.

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax

class CEWGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CEWGATConv, self).__init__(aggr='add', node_dim=0)  # 确保 node_dim 设置为 0
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.att = torch.nn.Parameter(torch.Tensor(1, out_channels))
        torch.nn.init.xavier_uniform_(self.att.data, gain=1.414)
        self.node_dim = 0  # 明确指定 node_dim

    def forward(self, x, edge_index, cew):
        x = self.lin(x)
        # 确保 cew 是二维张量，尺寸为 [N, 1]
        if cew.dim() == 1:
            cew = cew.unsqueeze(-1)
        return self.propagate(edge_index, x=x, cew=cew)

def message(self, x_i, x_j, cew_i, cew_j, index):
    # x_i, x_j: [E, out_channels]
    # cew_i, cew_j: [E, 1]
    print(f"cew_i size: {cew_i.size()}, cew_j size: {cew_j.size()}")
    alpha = F.leaky_relu((x_i * self.att).sum(dim=-1) + (x_j * self.att).sum(dim=-1))
    alpha = alpha * (cew_i + cew_j).squeeze(-1)  # 将 cew_i 和 cew_j 的最后一维压缩
    alpha = softmax(alpha, index)
    return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out
    
class HybridGNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_0, hidden_dim_1, output_dim, heads):
        super(HybridGNNModel, self).__init__()
        self.cew_gat_conv = CEWGATConv(input_dim, hidden_dim_0)
        self.transformer_conv = TransformerConv(hidden_dim_0, hidden_dim_1, heads=heads, concat=False)
        self.fc = torch.nn.Linear(hidden_dim_1, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cew = data.cew 
        x = F.relu(self.cew_gat_conv(x, edge_index, cew))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.transformer_conv(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return x
