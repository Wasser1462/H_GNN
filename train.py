# train.py
# Author: zyw
# Date: 2024-10-23
# Description: Trains the Hybrid GNN model using CEW values and contrastive learning.
import os
import torch
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import yaml
from node2vec import Node2Vec
import seaborn as sns
import logging
from model import HybridGNNModel
from label import calculate_infection_ability_parallel, select_key_nodes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    logging.info(f'Loaded config: {config}')

use_cpu = config.get('use_cpu', False)
if use_cpu:
    device = torch.device('cpu')
    logging.info('Training using CPU')
else:
    gpu_ids = config.get('gpu_ids', [0])
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}')
        logging.info(f'Training using GPUs: {gpu_ids}')
    else:
        device = torch.device('cpu')
        logging.warning('CUDA is not available. Training using CPU instead.')

train_path = config['train_path']
test_path = config['test_path']
output_dir = config['output_result']
cache_path = config.get('cache_path', 'cache')  
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_path, exist_ok=True) 

def read_adjacency_matrix(file_path):
    try:
        adj_matrix = pd.read_excel(file_path, header=None, index_col=None).values
        logging.info(f'Read adjacency matrix from {file_path} with shape {adj_matrix.shape}')
        return adj_matrix
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading the file {file_path}: {e}")
        return None

def build_graph(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        logging.info(f'Removing isolated nodes: {isolated_nodes}')
        G.remove_nodes_from(isolated_nodes)
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        logging.info(f'Removing self-loops: {self_loops}')
        G.remove_edges_from(self_loops)
    return G

def get_node_embeddings(G, config):
    node2vec = Node2Vec(
        G,
        dimensions=config['dimensions'],
        walk_length=config['walk_length'],
        num_walks=config['num_walks'],
        workers=config['workers']
    )
    model_n2v = node2vec.fit(
        window=config['window'],
        min_count=config['min_count'],
        epochs=config['node2vec_epoch']
    )
    node_embeddings = {node: model_n2v.wv[str(node)] for node in G.nodes()}
    embedding_matrix = [node_embeddings[node] for node in G.nodes()]
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
    logging.info(f'Embedding matrix shape: {embedding_matrix.shape}')
    return embedding_matrix

def compute_cew(adj_matrix, device):
    from connection_entropy_weights import calculate_cew
    cew_values = calculate_cew(adj_matrix)
    cew_values = torch.FloatTensor(cew_values).view(-1, 1).to(device)
    cew_min = cew_values.min()
    cew_max = cew_values.max()
    cew_norm = (cew_values - cew_min) / (cew_max - cew_min + 1e-10)
    return cew_norm

def create_data_object(G, embedding_matrix, cew_norm, labels, config):
    
    features = embedding_matrix
    logging.info(f'Features shape: {features.shape}')
    
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
    data = Data(x=features, edge_index=edge_index)
    data.y = labels
    data.cew = cew_norm
    
    return data

def process_graph(file_path, config):
    adj_matrix = read_adjacency_matrix(file_path)
    input_filename = os.path.basename(file_path).replace('.xlsx', '')
    if adj_matrix is None:
        return None
    G = build_graph(adj_matrix)
    if len(G) == 0:
        logging.warning(f'Graph from {file_path} has no nodes after removing isolated nodes.')
        return None
    cew_norm = compute_cew(adj_matrix, torch.device('cpu')) 
    
    cache_embeddings_path = os.path.join(cache_path, f"{input_filename}_embeddings.pt")
    cache_labels_path = os.path.join(cache_path, f"{input_filename}_labels.pt")
    
    if config['use_cache'] and os.path.exists(cache_embeddings_path) and os.path.exists(cache_labels_path):
        logging.info(f'Loading cached embeddings from {cache_embeddings_path} and labels from {cache_labels_path}.')
        embedding_matrix = torch.load(cache_embeddings_path, map_location='cpu',weights_only=True)  
        labels = torch.load(cache_labels_path, map_location='cpu',weights_only=True)  
    else:
        logging.info('Calculating node2vec embeddings')
        embedding_matrix = get_node_embeddings(G, config).to('cpu')  
        if config['use_cache']:
            torch.save(embedding_matrix, cache_embeddings_path)
            logging.info(f'Embeddings saved to cache: {cache_embeddings_path}')
        
        logging.info('Calculating infection ability and selecting key nodes')
        infection_ability = calculate_infection_ability_parallel(
            G,
            config['beta'],
            config['gamma'],
            config['steps'],
            config['num_simulations']
        ) 
        labels = select_key_nodes(G, infection_ability, config['key_node'])
        labels = torch.tensor(labels, dtype=torch.long).to('cpu')  
        if config['use_cache']:
            torch.save(labels, cache_labels_path)
            logging.info(f'Labels saved to cache: {cache_labels_path}')
    
    data = create_data_object(G, embedding_matrix, cew_norm, labels, config)
    return data

train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.xlsx')]

train_files = train_files[:5]
train_data_list = []
for file in train_files:
    data = process_graph(file, config)
    if data is not None:
        train_data_list.append(data)

if len(train_data_list) == 0:
    logging.error("No valid training data found.")
    exit(1)

test_data = process_graph(test_path, config)
if test_data is None:
    logging.error("No valid test data found.")
    exit(1)

train_loader = DataLoader(train_data_list,
                        batch_size=config['batch_size'], 
                        shuffle=config['shuffle'],
                        num_workers=config.get('num_workers'),
                        pin_memory=True
                        )
logging.info(f'Number of training graphs: {len(train_data_list)}')
logging.info(f'Test graph nodes: {test_data.num_nodes}')

input_dim = train_data_list[0].x.shape[1]
hidden_dim_0 = config['hidden_dim_0']
hidden_dim_1 = config['hidden_dim_1']
output_dim = config['output_dim']
num_heads = config['num_heads']
learning_rate = config['learning_rate']
# train_split_ratio = config['train_split_ratio'] 
early_stopping = config['early_stopping']
early_stopping_patience = config['early_stopping_patience']

torch.manual_seed(config['seed'])
model = HybridGNNModel(input_dim, hidden_dim_0, hidden_dim_1, output_dim, num_heads).to(device)

# zengyongwang NOTE: Multi-GPU data parallel computing is not supported at the moment.
if not use_cpu and len(gpu_ids) > 1 and torch.cuda.is_available():
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
temperature = config['temperature']
contrastive_weight = config['contrastive_weight']

def get_all_nodes_mask(data):
    return torch.arange(data.num_nodes, device=device)

test_mask = get_all_nodes_mask(test_data)

best_loss = float('inf')
patience_counter = 0
losses = []
model_state = None
test_metrics = {}
def contrastive_loss_fn(z1, z2, temperature, device):
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
    if drop_prob == 0:
        return edge_index
    mask = torch.rand(edge_index.size(1)) >= drop_prob
    return edge_index[:, mask]

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode=config['mode'], 
                                                       factor=config['factor'], 
                                                       patience=config['patience'], 
                                                       verbose=config['verbose'])

for epoch in range(config['num_epochs']):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = 0  

        if config['loss_type'] in ['contrastive', 'both']:
            edge_index_1 = dropout_edge(batch.edge_index, config['drop_edge_prob'])
            edge_index_2 = dropout_edge(batch.edge_index, config['drop_edge_prob'])
            data1 = Data(x=batch.x, edge_index=edge_index_1).to(device)
            data1.cew = batch.cew.to(device)
            data2 = Data(x=batch.x, edge_index=edge_index_2).to(device)
            data2.cew = batch.cew.to(device)
            z1 = model(data1)
            z2 = model(data2)
        
        if config['loss_type'] in ['cross_entropy', 'both']:
            logits = model(batch)
            train_mask = get_all_nodes_mask(batch)
            # train_mask = train_mask.to(device)  
            train_masked_logits = logits[train_mask]
            train_masked_labels = batch.y[train_mask]
            loss_supervised = criterion(train_masked_logits, train_masked_labels)
            loss += loss_supervised
            if config['loss_type'] == 'both':
                loss += contrastive_weight * contrastive_loss_fn(z1[train_mask], z2[train_mask], temperature, device)
        
        elif config['loss_type'] == 'contrastive':
            train_mask = get_all_nodes_mask(batch)
            loss_contrastive = contrastive_loss_fn(z1[train_mask], z2[train_mask], temperature, device)
            loss += contrastive_weight * loss_contrastive
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    if epoch % 2 == 0:
        logging.info(f'Epoch {epoch}/{config["num_epochs"]}, Loss: {avg_loss:.4f}')
    
    if config.get('early_stopping', False):
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if not config.get('use_cpu', False) and len(config.get('gpu_ids', [])) > 1 and torch.cuda.is_available():
                model_state = model.module.state_dict().copy()
            else:
                model_state = model.state_dict().copy()
            logging.info(f'Best loss updated to {best_loss:.4f} at epoch {epoch}')
        else:
            patience_counter += 1
            logging.info(f'No improvement in loss for {patience_counter} epochs')
        if patience_counter >= config.get('early_stopping_patience', 10):
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    else:
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not config.get('use_cpu', False) and len(config.get('gpu_ids', [])) > 1 and torch.cuda.is_available():
                model_state = model.module.state_dict().copy()
            else:
                model_state = model.state_dict().copy()

output_dir = config['output_result']
if model_state is not None:
    model.load_state_dict(model_state)
    torch.save(model_state, os.path.join(output_dir, 'model.pt'))
    logging.info(f'Model saved to {os.path.join(output_dir, "model.pt")}')
else:
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_last_epoch.pt'))
    logging.warning("Early stopping has not been triggered, and the model of the last epoch is saved.")

model.eval()
with torch.no_grad():
    logits = model(test_data.to(device))
    test_mask = get_all_nodes_mask(test_data)
    test_logits = logits[test_mask].cpu()
    test_labels = test_data.y[test_mask].cpu()
    predictions = torch.argmax(test_logits, dim=1)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)
    
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

if len(torch.unique(test_labels)) == 2:
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
else:
    logging.warning("ROC curve is not applicable for multi-class classification.")
