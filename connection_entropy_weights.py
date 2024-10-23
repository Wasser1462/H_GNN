# Author: zyw
# Date: 2024-10-23
# Description: Calculates Connection Entropy Weight (CEW), which helps identify key nodes that contribute to the structure and stability of the network 

import networkx as nx
import numpy as np
import torch
import pandas as pd


def compute_entropy(probabilities):
    probabilities = torch.FloatTensor(probabilities)
    entropy = torch.sum(-probabilities * torch.log2(probabilities + 1e-10))
    return entropy


def calculate_entropy(adj_matrix):
    entropies = []
    for i in range(adj_matrix.shape[0]):
        neighbors = adj_matrix[i]
        unique_neighbors, counts = np.unique(neighbors, return_counts=True)
        probabilities = counts / np.sum(counts)
        entropy = compute_entropy(probabilities)
        entropies.append(entropy)
    return entropies


def calculate_comprehensive_degree(G, degree_values):
    comprehensive_degrees = []
    for node in G.nodes:
        ego = nx.ego_graph(G, node, radius=2)
        num_nodes = len(ego.nodes)
        N = num_nodes - 1
        a = degree_values[node] / N if N != 0 else 0
        d = N - degree_values[node]
        Y = degree_values[node] + a * d
        comprehensive_degrees.append(Y)
    return comprehensive_degrees


def calculate_edge_weight(G, degree_values):
    edge_weights = []
    for node in G.nodes:
        ego = nx.ego_graph(G, node, radius=2)
        num_nodes = len(ego.nodes)
        N = num_nodes - 1
        a = degree_values[node] / N if N != 0 else 0
        k = degree_values[node]
        W = k * G.degree(node)  
        neighbors = G.neighbors(node)  
        for neighbor in neighbors:
            W += degree_values[neighbor]  
        edge_weights.append(W)
    return edge_weights

def calculate_c_value(G, node):
    ego = nx.ego_graph(G, node, radius=2)
    num_nodes = len(ego.nodes)
    N = num_nodes - 1

    node_degree = G.degree(node)
    c_sum = 0

    for neighbor in ego.nodes:
        if neighbor != node:
            neighbor_degree = G.degree(neighbor)

            u = neighbor_degree / N
            d = N - 1
            C = node_degree + u * d
            c_sum += C
    return c_sum
def calculate_cew(adj_matrix):

    G = nx.Graph(adj_matrix)
    degree_values = dict(G.degree())

    entropies = calculate_entropy(adj_matrix)
    comprehensive_degrees = calculate_comprehensive_degree(G, degree_values)
    edge_weights = calculate_edge_weight(G, degree_values)

    cew_values = [Y + W*H for Y, W, H in zip(comprehensive_degrees, edge_weights, entropies)]

    return cew_values


