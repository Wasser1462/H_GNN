# Author: zyw
# Date: 2024-12-04
# Description: This script is used to label the nodes in a graph based on SIR model.

import numpy as np
import random
import os
from tqdm import tqdm
import networkx as nx
import logging
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sir_iterate(G, p, r):
    nx.set_node_attributes(G, {n: G.nodes[n]['state'] for n in G.nodes()}, 'next_state')
    for n in G.nodes():
        if G.nodes[n]['state'] == 'S':
            k = sum(1 for neighbor in G.neighbors(n) if G.nodes[neighbor]['state'] == 'I')
            infection_prob = 1 - (1 - p) ** k
            if infection_prob >= random.uniform(0, 1):
                G.nodes[n]['next_state'] = 'I'

    for n in G.nodes():
        current_state = G.nodes[n]['state']
        next_state = G.nodes[n]['next_state']

        if current_state == 'S':
            G.nodes[n]['state'] = next_state
        elif current_state == 'I':
            if G.nodes[n]['counter'] is not None and G.nodes[n]['counter'] >= r:
                G.nodes[n]['state'] = 'R'
                G.nodes[n]['counter'] = None 
            else:
                G.nodes[n]['counter'] += 1
        elif current_state == 'R':
            pass
    return G

def sir_simulate(G, p, r, initially_infected, steps):
    nx.set_node_attributes(G, 'S', name='state')
    nx.set_node_attributes(G, 0, name='counter')

    for node in initially_infected:
        G.nodes[node]['state'] = 'I'
        G.nodes[node]['counter'] = 0  

    infected_set = set(initially_infected)

    step = 0
    while len([x for x, y in G.nodes(data=True) if y['state'] == 'I']) > 0 and step < steps:
        sir_iterate(G, p, r)
        current_infected = [x for x, y in G.nodes(data=True) if y['state'] == 'I']
        infected_set.update(current_infected)
        step += 1
    return len(infected_set)

def sir_simulate_time_series(G, p, r, initially_infected, steps):
    nx.set_node_attributes(G, 'S', name='state')
    nx.set_node_attributes(G, 0, name='counter')

    for node in initially_infected:
        G.nodes[node]['state'] = 'I'
        G.nodes[node]['counter'] = 0  

    infected_counts = [len(initially_infected)]

    while len([x for x, y in G.nodes(data=True) if y['state'] == 'I']) > 0 and len(infected_counts) <= steps:
        sir_iterate(G, p, r)
        current_infected = len([x for x, y in G.nodes(data=True) if y['state'] == 'I'])
        infected_counts.append(current_infected)

    while len(infected_counts) < steps + 1:
        infected_counts.append(0)

    return infected_counts

def simulate_node(args):
    G, beta, gamma, steps, node, num_simulations = args
    total_infected = 0
    for _ in range(num_simulations):
        infected = sir_simulate(G.copy(), beta, gamma, initially_infected=[node], steps=steps)
        total_infected += infected 
    average_infected = total_infected / num_simulations
    return (node, average_infected)

def calculate_infection_ability_parallel(G, beta, gamma, steps, num_simulations, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1) 

    infection_ability = {}
    args = [(G, beta, gamma, steps, node, num_simulations) for node in G.nodes()]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(simulate_node, args), total=len(args), desc="Calculating infection ability"))

    for node, avg_infected in results:
        infection_ability[node] = avg_infected

    avg_infected_values = list(infection_ability.values())
    logging.info(f"Average infected count - min: {min(avg_infected_values)}, max: {max(avg_infected_values)}, mean: {np.mean(avg_infected_values):.2f}, std: {np.std(avg_infected_values):.2f}")

    return infection_ability

def select_key_nodes(G, infection_ability, top_percent):
    sorted_nodes = sorted(infection_ability.keys(), key=lambda node: infection_ability[node], reverse=True)
    top_n = max(1, int(len(sorted_nodes) * top_percent))  

    binary_matrix = [0] * len(G.nodes())
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

    for node in sorted_nodes[:top_n]:
        index = node_to_index[node]
        binary_matrix[index] = 1

    #logging.info(f"Selected top {top_n} ({top_percent*100}%) key nodes out of {len(G.nodes())}.")
    return binary_matrix

# def save_binary_matrix(binary_matrix, output_dir, steps):
#     output_file = os.path.join(output_dir, f"binary_matrix_{steps}_steps.txt")
#     try:
#         with open(output_file, 'w') as f:
#             f.write(str(binary_matrix))
#         logging.info(f"Binary matrix saved to {output_file}")
#     except Exception as e:
#         logging.error(f"Error saving binary matrix: {e}")

def plot_infection_counts(infection_ability, steps, output_dir):
    import matplotlib.pyplot as plt
    import os
    node_ids = list(infection_ability.keys())
    avg_infected_counts = [infection_ability[node] for node in node_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(node_ids, avg_infected_counts, color='skyblue')
    plt.xlabel('Node ID')
    plt.ylabel('Average Infected Nodes')
    plt.title(f'Average Infected Nodes per Node for {steps} Steps')
    plt.xticks(ticks=range(0, len(node_ids), max(1, len(node_ids)//20)), rotation=90) 
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'infection_counts_{steps}_steps.png')
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Infection counts bar plot saved to {output_file}")

def plot_infection_time_series(G, beta, gamma, steps, num_simulations, output_dir, sample_nodes=5):
    import matplotlib.pyplot as plt
    import random
    import os
    sampled_nodes = random.sample(list(G.nodes()), min(sample_nodes, len(G.nodes())))
    plt.figure(figsize=(12, 6))
    for node in sampled_nodes:
        total_infected_over_time = []
        for _ in range(num_simulations):
            infected = sir_simulate_time_series(G.copy(), beta, gamma, initially_infected=[node], steps=steps)
            total_infected_over_time.append(infected)
        avg_infected_over_time = np.mean(total_infected_over_time, axis=0)
        plt.plot(avg_infected_over_time, label=f'Node {node}')
    plt.xlabel('Step')
    plt.ylabel('Number of Infected Nodes')
    plt.title(f'Infected Nodes Over Time for {steps} Steps (Sampled Nodes)')
    plt.legend()
    plt.grid(True)
    output_file = os.path.join(output_dir, f'infected_nodes_time_series_{steps}_steps.png')
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Infection time series plot saved to {output_file}")

