import matplotlib.pyplot as plt
import numpy as np
import main as main
import seaborn as sns
import graph_classification as gc
from bgm import graph_edit_distance_bipartite, bgm_matrix
import networkx as nx
import pandas as pd
import networkx as nx

# Assuming cost_matrix is your matrix
def plot_cost_matrix(cost_matrix, title="Cost Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cost_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
    plt.xlabel('Graph 1 Nodes')
    plt.ylabel('Graph 2 Nodes')
    plt.title(title)
    plt.show()

def node_match(n1, n2):
    """
    * checks if two nodes are matching based on their labels

    * param n1: a dictionary representing the first node
    * param n2: a dictionary representing the second node

    * return: True if the nodes are matching, False otherwise
    """
    return n1['label'] == n2['label']

def edge_match(e1, e2):
    """
    * checks if two edges are matching based on their labels

    * param e1: a dictionary representing the first edge
    * param e2: a dictionary representing the second edge

    * return: True if the edges are matching, False otherwise
    """
    return e1['label'] == e2['label']

def has_duplicates(lst):
    return len(lst) != len(set(lst))

def load_data():
    dataset = "MUTAG"
    edges = np.loadtxt(f"data/{dataset}/{dataset}_A.txt", dtype=int, delimiter=',')
    edge_labels = np.loadtxt(f"data/{dataset}/{dataset}_edge_labels.txt", dtype=int)
    graph_indicator = np.loadtxt(f"data/{dataset}/{dataset}_graph_indicator.txt", dtype=int)
    graph_labels = np.loadtxt(f"data/{dataset}/{dataset}_graph_labels.txt", dtype=int)
    node_labels = np.loadtxt(f"data/{dataset}/{dataset}_node_labels.txt", dtype=int)
    #node_attributes = np.loadtxt(f"{dataset}/{dataset}_node_attributes.txt", dtype=float, delimiter=',')

    number_of_graphs = max(graph_indicator)
    graphs = [nx.Graph() for i in range(number_of_graphs)]
    #for i in range(number_of_graphs):
    #   graphs.append(nx.Graph())

    node_mapping = {}

    #Knoten erstellen
    for i in range(len(node_labels)):
        graph_id = graph_indicator[i] - 1
        graph = graphs[graph_id]
        node_graph_id = graph.number_of_nodes()
        graph.add_node(node_graph_id, label = node_labels[i])
        node_mapping[i+1] = (graph_id, node_graph_id)

    #Kanten erstellen
    for i, (n1, n2) in enumerate(edges):        
        graph_id1, node_graph_id1 = node_mapping[n1]
        graph_id2, node_graph_id2 = node_mapping[n2]

        if graph_id1 == graph_id2:
            graphs[graph_id1].add_edge(node_graph_id1, node_graph_id2, label = edge_labels[i])

    for i in range(len(graphs)):
        graphs[i].graph["label"] = graph_labels[i] 

    return graphs


if __name__ == "__main__":

    n = 40

    graphs = main.load_graphs("MUTAG", n)

    cost_matrix, edit_matrix, matchings = main.calculate_cost_matrix(graphs, 5, 0)


    # for i in range(n): 
    #     for j in range(n):
    #         if i == j:
    #             continue
    #         if i > j:
    #             continue
    #         # check the edit path for doubling entries
    #         if has_duplicates(edit_matrix[i,j]):
    #             print(f"Graph {i+1} and Graph {j+1} have double entries in the edit path")
    #         new_graph = main.graph_matcher(graphs[i+1], graphs[j+1], edit_matrix[i,j], matchings[i,j])
    #         current_bool = main.isomorph_check(graphs[j+1], new_graph)
    #         if not current_bool:
    #             print(edit_matrix[i,j])
    #             print(matchings[i,j])
    #             main.print_two_graphs(graphs[j+1], new_graph)
    #             print(i+1, j+1)
    #             print("\n")
    #             print("\n")

    bgm_cost_matrix = main.nx_cost_matrix(graphs, n_iter=0)
    # bgm_cost_matrix = main.cnt_bound_matrix(graphs, cnt_matrix=cost_matrix)
    # bgm_cost_matrix = bgm_matrix(graphss)

    # print(bgm_cost_matrix)

    # avg value of cost_matrix
    print(f"Mean")
    print(np.mean(cost_matrix))
    print(np.mean(bgm_cost_matrix))
    print("--------------------")
    print(f"Average")
    print(np.average(cost_matrix))
    print(np.average(bgm_cost_matrix))
    print("--------------------")
    print(f"Max. Difference")
    # highest difference in cost_matrix
    # find min, not on diagonal
    print(np.max(cost_matrix) - np.min(cost_matrix[np.nonzero(cost_matrix)]))
    print(np.max(bgm_cost_matrix) - np.min(bgm_cost_matrix[np.nonzero(bgm_cost_matrix)]))
    print("--------------------")

    # for i in range(n): 
    #     for j in range(n):
    #         if i == j:
    #             continue
    #         if i > j:
    #             continue
    #         if has_duplicates(bgm_edit_matrix[i,j]):
    #             print(f"Graph {i+1} and Graph {j+1} have double entries in the edit path")
    #         new_graph = main.graph_matcher(graphs[i+1], graphs[j+1], bgm_edit_matrix[i,j], bgm_matchings[i,j])
    #         current_bool = main.isomorph_check(graphs[j+1], new_graph)
    #         if not current_bool:
    #             print(bgm_edit_matrix[i,j])
    #             print(bgm_matchings[i,j])
    #             main.print_two_graphs(graphs[j+1], new_graph)
    #             print(i+1, j+1)
    #             print("\n")
    #             print("\n")

    # save cost matrix to file
    # np.savetxt(f"{dataset_name}_cost_matrix.csv", cost_matrix, delimiter=",")

    print("Done")
