# train.py
# Author: zyw
# Date: 2024-10-23
# Description: Trains the Hybrid GNN model using CEW values and contrastive learning.

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from multiprocessing import Pool, cpu_count
from node2vec import Node2Vec
import seaborn as sns
from connection_entropy_weights import calculate_cew, calculate_c_value
import logging
from model import HybridGNNModel
from label import select_key_nodes,calculate_infection_ability_parallel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Training using: {"GPU" if torch.cuda.is_available() else "CPU"}')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    logging.info(f'Loaded config: {config}')

file_path = config['file_path']
output_dir = config['output_result']
os.makedirs(output_dir, exist_ok=True)

try:
    adj_matrix = pd.read_excel(file_path, header=None, index_col=None).values
except FileNotFoundError:
    logging.error(f"File not found: {file_path}")
    exit(1)

G = nx.Graph(adj_matrix)
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)
G.remove_edges_from(nx.selfloop_edges(G))

cew_values = calculate_cew(adj_matrix)
cew_values = torch.FloatTensor(cew_values).view(-1, 1).to(device)

cew_min = cew_values.min()
cew_max = cew_values.max()
cew_norm = (cew_values - cew_min) / (cew_max - cew_min + 1e-10)

node2vec = Node2Vec(G, dimensions=config['dimensions'], walk_length=config['walk_length'],
                    num_walks=config['num_walks'], workers=config['workers'])
model_n2v = node2vec.fit(window=config['window'], min_count=config['min_count'], epochs=config['node2vec_epoch'])
node_embeddings = {node: model_n2v.wv[str(node)] for node in G.nodes}
embedding_matrix = [node_embeddings[node] for node in G.nodes]
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float).to(device)
logging.info(f'Embedding matrix shape: {embedding_matrix.shape}')

#features = torch.cat([embedding_matrix, cew_norm], dim=1)
features = embedding_matrix
logging.info(f'Features shape: {features.shape}')
infection_ability = calculate_infection_ability_parallel(G, config['beta'], config['gamma'], config['steps'], config['num_simulations'], num_workers=cpu_count()-1)
binary_matrix = select_key_nodes(G, infection_ability, config['key_node'])
logging.info(f'label matrix : {binary_matrix}')

edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().to(device)
data = Data(x=features, edge_index=edge_index)
data.y = torch.LongTensor(binary_matrix).to(device)
data.cew = cew_norm

input_dim = features.shape[1]
hidden_dim_0 = config['hidden_dim_0']
hidden_dim_1 = config['hidden_dim_1']
output_dim = config['output_dim']
num_heads = config['num_heads']
learning_rate = config['learning_rate']
train_split_ratio = config['train_split_ratio']
early_stopping = config['early_stopping']
early_stopping_patience = config['early_stopping_patience']

torch.manual_seed(config['seed'])
model = HybridGNNModel(input_dim, hidden_dim_0, hidden_dim_1, output_dim, num_heads).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

temperature = config['temperature']
contrastive_weight = config['contrastive_weight']

mask = torch.randperm(len(G.nodes))
train_split = int(len(G.nodes) * train_split_ratio)
train_mask = mask[:train_split].to(device)
test_mask = mask[train_split:].to(device)

best_loss = float('inf')
patience_counter = 0
losses = []

model_state = None
test_metrics = {}

def contrastive_loss(z1, z2, temperature):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    positives = torch.sum(z1 * z2, dim=1)
    negatives = torch.mm(z1, z2.t())
    mask = torch.eye(batch_size).to(device)
    negatives = negatives.masked_select(~mask.bool()).view(batch_size, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(device)
    loss = F.cross_entropy(logits / temperature, labels)
    return loss

def dropout_edge(edge_index, drop_prob):
    mask = torch.rand(edge_index.size(1)) >= drop_prob
    return edge_index[:, mask]

for epoch in range(config['num_epochs']):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = edge_index
    edge_index_2 = edge_index

    drop_prob = config['drop_edge_prob']
    edge_index_1 = dropout_edge(edge_index, drop_prob)
    edge_index_2 = dropout_edge(edge_index, drop_prob)

    data1 = Data(x=features, edge_index=edge_index_1)
    data1.cew = cew_norm  

    data2 = Data(x=features, edge_index=edge_index_2)
    data2.cew = cew_norm  

    z1 = model(data1)
    z2 = model(data2)
    logits = model(data)

    train_masked_logits = logits[train_mask]
    train_masked_labels = data.y[train_mask]
    loss_supervised = criterion(train_masked_logits, train_masked_labels)

    loss_contrastive = contrastive_loss(z1[train_mask], z2[train_mask], temperature)

    loss = loss_supervised + contrastive_weight * loss_contrastive
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 5 == 0:
        logging.info(f'Epoch {epoch}/{config["num_epochs"]}, Loss: {loss.item():.4f}')

    if early_stopping:
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
else:
    if loss.item() < best_loss:
        best_loss = loss.item()
        model_state = model.state_dict().copy()

if model_state is not None:
    model.load_state_dict(model_state)
    torch.save(model_state, os.path.join(output_dir, 'model.pt'))
else:
    model_state = model.state_dict().copy()
    torch.save(model_state, os.path.join(output_dir, 'model_last_epoch.pt'))
    logging.warning("Early stopping has not been triggered, and the model of the last epoch is saved.")

model.eval()
with torch.no_grad():
    logits = model(data)
    test_logits = logits[test_mask]
    test_labels = data.y[test_mask]
    predictions = torch.argmax(test_logits, dim=1)
    accuracy = accuracy_score(test_labels.cpu(), predictions.cpu())
    precision = precision_score(test_labels.cpu(), predictions.cpu(), average='macro', zero_division=0)
    recall = recall_score(test_labels.cpu(), predictions.cpu(), average='macro', zero_division=0)
    f1 = f1_score(test_labels.cpu(), predictions.cpu(), average='macro', zero_division=0)

    test_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

logging.info("Model Evaluation on Test Set:")
logging.info(f"Accuracy: {test_metrics['accuracy']}")
logging.info(f"Precision: {test_metrics['precision']}")
logging.info(f"Recall: {test_metrics['recall']}")
logging.info(f"F1 Score: {test_metrics['f1']}")


plt.figure()
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'))
plt.close()


conf_matrix = confusion_matrix(test_labels.cpu(), predictions.cpu())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

fpr, tpr, _ = roc_curve(test_labels.cpu(), test_logits[:, 1].detach().cpu().numpy())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()
