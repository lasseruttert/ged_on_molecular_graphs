import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time as t
from collections import defaultdict
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    dataset = "AIDS"
    edges = np.loadtxt(f"{dataset}/{dataset}_A.txt", dtype=int, delimiter=',')
    edge_labels = np.loadtxt(f"{dataset}/{dataset}_edge_labels.txt", dtype=int)
    graph_indicator = np.loadtxt(f"{dataset}/{dataset}_graph_indicator.txt", dtype=int)
    graph_labels = np.loadtxt(f"{dataset}/{dataset}_graph_labels.txt", dtype=int)
    node_labels = np.loadtxt(f"{dataset}/{dataset}_node_labels.txt", dtype=int)
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

def draw_cost_matrix(cost_matrix):
    plt.figure(figsize=(16, 16))
    plt.imshow(cost_matrix, cmap="Blues", aspect="auto")
    row_labels = [f"R {i}" for i in range(cost_matrix.shape[0])]
    col_labels = [f"C {i}" for i in range(cost_matrix.shape[1])]
    plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels)
    plt.yticks(ticks=np.arange(len(row_labels)), labels=row_labels)
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            plt.text(j, i, str(cost_matrix[i, j]), ha="center", va="center", color="black")
    plt.show()

def get_free_node_id(graph):
    existing_ids = set(graph.nodes) 
    node_id = 0
    
    while node_id in existing_ids:
        node_id += 1
        
    return node_id

def edge_exists(edge, edge_set):
    u, v = edge
    return (u, v) in edge_set or (v, u) in edge_set

def edge_has_unmatched_node(edge, matching):
    u, v = edge
    return u not in matching or v not in matching 

def print_graph(graph):   
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=False, node_size=500, node_color='lightblue', font_size=12)
    node_labels = nx.get_node_attributes(graph, 'label')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=12)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    plt.show()

#Abschätzung der Kosten von Kantenoperationen durch eine Knotensubstitution impliziert
def edge_cost_matrix(graph1, graph2, matching):
    e1, e2 = list(graph1.edges), list(graph2.edges)
    n1, n2 = len(graph1.edges), len(graph2.edges)
    size = n1 + n2
    cost_matrix = np.full((size, size), fill_value=10000)
    graph_edited = graph1.copy()
    total_cost = 0
    #e1p = []
    #e2p = []
    for i in range(n1):
        for j in range(n2):                
            ids1 = e1[i]
            ids1_matching = (matching.get(ids1[0]), matching.get(ids1[1]))
            ids2 = e2[j]
            if(ids1_matching[0] == None or ids1_matching[1] == None):
                #if(ids1 not in e1p):
                    #total_cost += 1
                    #e1p.append(ids1)
                    continue
            elif(ids2[0] not in matching.values() or ids2[1] not in matching.values()):
                #if(ids2 not in e2p):
                    #total_cost += 1
                    #e2p.append(ids2)
                    continue
            else:    
                ids1_matchingint = ((matching.get(ids1[0])), int(matching.get(ids1[1])))
                if ids1_matchingint == ids2:
                    #isomorph
                    if graph1[e1[i][0]][e1[i][1]]["label"] == graph2[e2[j][0]][e2[j][1]]["label"]:
                        cost_matrix[i, j] = 0
                    else:
                        #Kantensubstitution
                        cost_matrix[i, j] = 1
                else:
                    cost_matrix[i, j] = 1000

    for i in range(n2, size):
        cost_matrix[i-n2, i] = 1

    for j in range(n1, size):
        cost_matrix[j, j-n1] = 1 

    for i in range(n2, size):
        for j in range(n1, size):
             cost_matrix[j, i] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost += cost_matrix[row_ind, col_ind].sum()
    print(total_cost)
    plt.figure(figsize=(16, 16))
    plt.imshow(cost_matrix, cmap="Blues", aspect="auto")
    row_labels = [f"R {i}" for i in range(cost_matrix.shape[0])]
    col_labels = [f"C {i}" for i in range(cost_matrix.shape[1])]
    plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels)
    plt.yticks(ticks=np.arange(len(row_labels)), labels=row_labels)
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            plt.text(j, i, str(cost_matrix[i, j]), ha="center", va="center", color="black")
    plt.show()
    return total_cost

def compute_cost_matrix(graph1, graph2):
    n1, n2 = len(graph1.nodes), len(graph2.nodes)
    size = n1 + n2
    cost_matrix = np.full((size, size), fill_value=10000)  

    #Knotensubstitutionen
    for i, u in enumerate(graph1.nodes):
        for j, v in enumerate(graph2.nodes):
            cost_matrix[i, j] = node_edit_cost(graph1.nodes[u]["label"], graph2.nodes[v]["label"])

    #Löschoperationen
    for i in range(n2, size):
        cost_matrix[i-n2, i] = 1

    #Einfügeoperationen
    for j in range(n1, size):
        cost_matrix[j, j-n1] = 1 

    #Operationen auf Dummy-Knoten auf null setzen
    for i in range(n2, size):
        for j in range(n1, size):
             cost_matrix[j, i] = 0

    return cost_matrix

def node_edit_cost(label1, label2):
    return 0 if label1 == label2 else 1

def graph_edit_distance_bipartite(graph1, graph2):
    cost_matrix = compute_cost_matrix(graph1, graph2)
    draw_cost_matrix(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    matching = []
    for i,j in zip(row_ind, col_ind):
        if i < len(graph1.nodes) and j < len(graph2.nodes):
            matching.append((i,j))
        elif i < len(graph1.nodes) and j >= len(graph2.nodes):
            #total_cost += graph1.degree(j-len(graph2.nodes))
            matching.append((i,None))
    
    for i,j in zip(row_ind, col_ind):
        if i >= len(graph1.nodes) and j < len(graph2.nodes):
            new_id = get_free_node_id(graph1)
            matching.append((new_id, j))

    matching = dict(matching)
    total_cost += edge_cost_matrix(graph1, graph2, matching)
    return total_cost

graphs = load_data()

def node_match(n1, n2):
    return n1["label"] == n2["label"]

def edge_match(e1, e2):
    return e1["label"] == e2["label"]

def node_ins_cost(n):
    return 1

def node_del_cost(n):
    return 1

def node_subst_cost(n1, n2):
    return 0 if n1["label"] == n2["label"] else 1

def edge_ins_cost(e):
    return 1

def edge_del_cost(e):
    return 1

def edge_subst_cost(e1, e2):
    return 0 if e1["label"] == e2["label"] else 1

def runtime_beam_bipartite():
    runtime_beam = []
    runtime_bipartite = []
    for a in range(int(len(graphs))):
        for b in range(int(len(graphs))):
            graph1 = graphs[a]
            graph2 = graphs[b]
            size = len(graph1.nodes) + len(graph1.edges) + len(graph2.nodes) + len(graph2.edges)
            time = t.time()
            bipartite = graph_edit_distance_bipartite(graph1, graph2)
            bi_time = t.time() - time
            time = t.time()
            #beam = next(nx.optimize_graph_edit_distance(graph1, graph2, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost, node_subst_cost=node_subst_cost, edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost))
            beam =  nx.graph_edit_distance(graph1, graph2,upper_bound=bipartite, timeout=0.1, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost, node_subst_cost=node_subst_cost, edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost)
            beam_time = t.time() - time
            print(bipartite, beam)
            runtime_beam.append((size, beam_time))
            runtime_bipartite.append((size, bi_time))
            #print(bipartite, beam, bi_time, beam_time, a, b)
    #print(np.mean(runtime_bipartite), np.mean(runtime_beam))

    grouped_runtimes = defaultdict(list)
    for size, runtime in runtime_beam:
        grouped_runtimes[size].append(runtime)

    average_runtimes = {size: sum(runtimes) / len(runtimes) for size, runtimes in grouped_runtimes.items()}

    sizes = list(average_runtimes.keys())
    averages = list(average_runtimes.values())

    plt.bar(sizes, averages, color='b')

    plt.xlabel('Graphgröße')
    plt.ylabel('Durchschnittliche Laufzeit (Sekunden)')
    plt.title('Durchschnittliche Laufzeit in Abhängigkeit von der Graphgröße')

    plt.show()
    grouped_runtimes_bipartite = defaultdict(list)
    for size, runtime in runtime_bipartite:
        grouped_runtimes_bipartite[size].append(runtime)

    average_runtimes_bipartite = {size: sum(runtimes) / len(runtimes) for size, runtimes in grouped_runtimes_bipartite.items()}

    sizes_bipartite = list(average_runtimes_bipartite.keys())
    averages_bipartite = list(average_runtimes_bipartite.values())

    plt.bar(sizes_bipartite, averages_bipartite, color='b')

    plt.xlabel('Graphgröße')
    plt.ylabel('Durchschnittliche Laufzeit (Sekunden) - Bipartite')
    plt.title('Durchschnittliche Laufzeit in Abhängigkeit von der Graphgröße - Bipartite')

    plt.show()

def knn_graph_classification(query_graph, train_graphs, bipartite=True):
    distances = []
    for train_graph in train_graphs:
        ged = graph_edit_distance_bipartite(query_graph, train_graph) if bipartite else next(nx.optimize_graph_edit_distance(query_graph, train_graph, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost, node_subst_cost=node_subst_cost, edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost))
        distances.append(ged)
    
    nearest_neighbor_index = np.argmin(distances)
    return train_graphs[nearest_neighbor_index].graph["label"]

def test_knn(bipartite=True):
    test_graphs = graphs[0::4]
    train_graphs = [graph for graph in graphs if graph not in test_graphs]
    #test_graphs = test_graphs[0::10]
    tp, fp, tn, fn = 0, 0 ,0 ,0
    start_time = t.time()
    for i in range(len(test_graphs)):
        classification = knn_graph_classification(test_graphs[i], train_graphs, bipartite)
        test_label = test_graphs[i].graph["label"]
        if classification == 1:
            if test_label == 1:
                tp += 1
            else:
                fp += 1
        elif classification == -1 or classification == 0:
            if test_label == -1 or test_label == 0:
                tn += 1
            else:
                fn += 1

    return tp, fp, tn, fn, len(test_graphs), (t.time() - start_time)

#tp, fp, tn, fn, length, runtime = test_knn()
#print(tp, fp)
#print(fn, tn)
#print(f"Average Runtime: {runtime/length}")
#tp, fp, tn, fn, length, runtime = test_knn(bipartite=False)
#print(tp, fp)
#print(fn, tn)
#print(f"Average Runtime: {runtime/length}")
print_graph(graphs[90])
print_graph(graphs[5])
print(graph_edit_distance_bipartite(graphs[90], graphs[5]))