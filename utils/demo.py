import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_adjacency_matrix(file_path):
    df = pd.read_excel(file_path, header=None)
    adj_matrix = df.values
    return adj_matrix


def adjacency_to_edge_index(adj_matrix):
    edge_index = np.array(np.nonzero(adj_matrix)) 
    return torch.tensor(edge_index, dtype=torch.long)


def create_node_features(num_nodes):
    features = np.ones((num_nodes, 1)) 
    return torch.tensor(features, dtype=torch.float)

def create_graph_data(file_path):
    adj_matrix = load_adjacency_matrix(file_path)
    edge_index = adjacency_to_edge_index(adj_matrix)
    num_nodes = adj_matrix.shape[0]
    x = create_node_features(num_nodes)

    y = torch.randint(0, 2, (num_nodes,)) 
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

train_data = []
train_files = ['train1.xlsx', 'train2.xlsx', 'train3.xlsx', 'train4.xlsx', 'train5.xlsx']

for file in train_files:
    data = create_graph_data(f'./train/{file}')
    train_data.append(data)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

test_data = create_graph_data('./test/test.xlsx')
test_loader = DataLoader([test_data], batch_size=1)

for batch in train_loader:
    print(batch)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(input_dim=1, hidden_dim=16, output_dim=2)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train(model, loader):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        print('Training loss:', loss.item())

train(model, train_loader)

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        out = model(batch)
        _, predicted = torch.max(out, 1)
        correct += (predicted == batch.y).sum().item()
        total += batch.y.size(0)
    accuracy = correct / total
    print('Test Accuracy:', accuracy)

test(model, test_loader)
