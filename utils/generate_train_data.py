# Author: zyw
# Date: 2024-12-06
# Description: 

import os
import networkx as nx
import pandas as pd

def edgelist_to_adjacency_matrix(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in filter(lambda f: f.endswith(".edgelist"), os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f'train_{file_name.replace(".edgelist", ".xlsx")}')
        
        G = nx.read_edgelist(input_path, data=True)
        adj_matrix = nx.to_pandas_adjacency(G, dtype=float)
        
        adj_matrix.to_excel(output_path)
        print(f"Saved adjacency matrix to: {output_path}")

input_folder = "/data1/zengyongwang/test/H_GNN/data/network_edgelist"
output_folder = "/data1/zengyongwang/test/H_GNN/data"
edgelist_to_adjacency_matrix(input_folder, output_folder)
