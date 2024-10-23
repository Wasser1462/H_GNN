# Author: zyw
# Date: 2024-10-23
# Description: This script trains a GNN model using Connection Entropy Weight (CEW) values as features for the training process.

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import GNNModel
import yaml
from node2vec import Node2Vec
import seaborn as sns
from connection_entropy_weights import calculate_cew, calculate_c_value

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

file_path = config['file_path']

try:
    adj_matrix = pd.read_excel(file_path, header=None, index_col=None).values
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

cew_values = calculate_cew(adj_matrix)
cew_values = torch.FloatTensor(cew_values)  
cew_values = cew_values.view(-1, 1) 

G = nx.Graph(adj_matrix)
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)
G.remove_edges_from(nx.selfloop_edges(G))

node2vec = Node2Vec(G, dimensions=config['dimensions'], walk_length=config['walk_length'],
                    num_walks=config['num_walks'], workers=config['workers'])
model = node2vec.fit(window=config['window'], min_count=config['min_count'])
node_embeddings = {node: model.wv[node] for node in G.nodes}
embedding_matrix = [node_embeddings[node] for node in G.nodes]
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

features = torch.cat((embedding_matrix, cew_values), dim=1)

binary_matrix = np.zeros((len(G.nodes),), dtype=int)
C_values = {}
with tqdm(total=len(G.nodes), desc="Calculating C values") as pbar:
    for node in G.nodes:
        C_values[node] = calculate_c_value(G, node)
        pbar.update(1)

sorted_nodes = sorted(C_values.keys(), key=lambda node: C_values[node], reverse=True)
top_percent = int(len(sorted_nodes) * 0.3)
node_to_index = {node: idx for idx, node in enumerate(G.nodes)}
for node in sorted_nodes[:top_percent]:
    index = node_to_index[node]
    binary_matrix[index] = 1

data = Data(x=features, edge_index=torch.tensor(np.where(adj_matrix == 1), dtype=torch.long),
            y=torch.LongTensor(binary_matrix))

input_dim = features.shape[1]
hidden_dim_0 = config['hidden_dim_0']
hidden_dim_1 = config['hidden_dim_1']
output_dim = config['output_dim']
num_heads = config['num_heads']
learning_rate = config['learning_rate']

torch.manual_seed(config['seed'])
mask = torch.randperm(len(G.nodes))
train_split = int(len(G.nodes) * config['train_split_ratio'])
train_mask = mask[:train_split]
test_mask = mask[train_split:]

top_models = []
num_best_models = 5

losses = []
accuracies = []

for epoch in range(config['num_epochs']):
    model = GNNModel(input_dim, hidden_dim_0, hidden_dim_1, output_dim, num_heads)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    train_masked_logits = logits[train_mask]
    train_masked_labels = data.y[train_mask]
    loss = criterion(train_masked_logits, train_masked_labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    model.eval()
    logits = model(data)
    test_logits = logits[test_mask]
    test_labels = data.y[test_mask]
    predictions = torch.argmax(test_logits, dim=1)
    accuracy = accuracy_score(test_labels, predictions)
    accuracies.append(accuracy)
    

    if len(top_models) < num_best_models:
        top_models.append((accuracy, model.state_dict()))
    else:
        min_accuracy = min(top_models, key=lambda x: x[0])[0]
        if accuracy > min_accuracy:
            top_models = sorted(top_models, key=lambda x: x[0], reverse=True)
            top_models[-1] = (accuracy, model.state_dict())
    
    if epoch % 5 == 0:
        print(f'Epoch {epoch}/{config["num_epochs"]}, Loss: {loss.item()}, Accuracy: {accuracy}')


avg_state_dict = None
for i, (_, state_dict) in enumerate(top_models):
    if avg_state_dict is None:
        avg_state_dict = {key: value.clone() for key, value in state_dict.items()}
    else:
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

for key in avg_state_dict:
    avg_state_dict[key] /= len(top_models)

avg_model = GNNModel(input_dim, hidden_dim_0, hidden_dim_1, output_dim, num_heads)
avg_model.load_state_dict(avg_state_dict)
torch.save(avg_model.state_dict(), 'model.pt')


avg_model.eval()
logits = avg_model(data)
test_logits = logits[test_mask]
test_labels = data.y[test_mask]
predictions = torch.argmax(test_logits, dim=1)
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
print("Final Averaged Model Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


plt.figure()
plt.plot(range(1, config['num_epochs'] + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

conf_matrix = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(test_labels, test_logits[:, 1].detach().numpy())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
