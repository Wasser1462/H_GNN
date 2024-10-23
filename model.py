# Author: zyw
# Date: 2024-10-23
# Description: A GNN model with attention mechanisms that aggregates node features based on their neighbors, enabling effective graph representation learning.

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_0, hidden_dim_1, output_dim, num_heads=1):
        super(GNNModel, self).__init__()
        
        self.num_heads = num_heads

        self.W0 = nn.Parameter(torch.Tensor(num_heads, input_dim, hidden_dim_0))
        self.b0 = nn.Parameter(torch.Tensor(num_heads, hidden_dim_0))
        
        self.W1 = nn.Parameter(torch.Tensor(num_heads, hidden_dim_0, hidden_dim_1))
        self.b1 = nn.Parameter(torch.Tensor(num_heads, hidden_dim_1))
        
        self.W2 = nn.Parameter(torch.Tensor(hidden_dim_1 * num_heads, output_dim))
        self.b2 = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_heads):
            nn.init.xavier_uniform_(self.W0[i])
            nn.init.zeros_(self.b0[i])
            nn.init.xavier_uniform_(self.W1[i])
            nn.init.zeros_(self.b1[i])

        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b2)

    def attention_mechanism(self, x, edge_index):
        row, col = edge_index
        edge_weight = torch.zeros_like(row, dtype=torch.float)
        
        for i, (r, c) in enumerate(zip(row, col)):
            score = torch.dot(x[r], x[c])
            edge_weight[i] = F.leaky_relu(score, negative_slope=0.2)
        
        edge_weight = F.softmax(edge_weight, dim=0)
        
        new_x = torch.zeros_like(x)
        for i, (r, c) in enumerate(zip(row, col)):
            new_x[r] += edge_weight[i] * x[c]
        
        return new_x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        all_heads = []
        for i in range(self.num_heads):
            head_x = torch.matmul(x, self.W0[i]) + self.b0[i]
            head_x = F.relu(head_x)
            head_x = self.attention_mechanism(head_x, edge_index)
            all_heads.append(head_x)
        
        x = torch.cat(all_heads, dim=-1)

        all_heads = []
        for i in range(self.num_heads):
            head_x = torch.matmul(x, self.W1[i]) + self.b1[i]
            head_x = F.relu(head_x)
            head_x = self.attention_mechanism(head_x, edge_index)
            all_heads.append(head_x)

        x = torch.cat(all_heads, dim=-1)

        x = torch.matmul(x, self.W2) + self.b2
        return x
