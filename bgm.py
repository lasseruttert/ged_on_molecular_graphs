import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time as t
from collections import defaultdict
import os
from main import sdted, build_nt, create_subgraph_dict, calculate_costs

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
    return total_cost

def compute_cost_matrix(graph1, graph2):
    n1, n2 = len(graph1.nodes), len(graph2.nodes)
    size = n1 + n2
    cost_matrix = np.full((size, size), fill_value=10000)  
    cache = {}

    #Knotensubstitutionen
    for i, u in enumerate(graph1.nodes):
        for j, v in enumerate(graph2.nodes):
            nt1 = build_nt(graph1, u, 5, 0)
            nt2 = build_nt(graph2, v, 5, 0)
            nt1 = calculate_costs(nt1)
            nt2 = calculate_costs(nt2)
            nt1_subgraphs = create_subgraph_dict(nt1)
            nt2_subgraphs = create_subgraph_dict(nt2)
            cost_matrix[i, j] = sdted(nt1, nt2, nt1_subgraphs, nt2_subgraphs, cache)

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

import concurrent.futures
from itertools import combinations

def bgm_matrix(graphs):
    """
    * calculates the GED cost matrix between a set of graphs using the Hungarian Algorithm based on the standard cost function

    * param graphs: a dictionary containing networkx Graph objects representing the graphs
    
    * return: the GED cost matrix between the graphs, the edit paths and the matchings between the graphs

    * description:
    * The function calculates the GED cost matrix between a set of graphs using the Hungarian Algorithm based on the standard cost function
    """
    basetime = t.time()

    # create the cost matrix, edit paths and matchings
    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0)

    # use concurrent.futures to parallelize the calculation of the GED cost matrix
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        # calculate the GED between all pairs of graphs
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(graph_edit_distance_bipartite, graphs[graph_ids[i]], graphs[graph_ids[j]])

        # get the results of the futures and store them in the cost matrix, edit paths and matchings
        for (i, j), future in futures.items():
            min_GED = future.result()
            cost_matrix[i, j] = cost_matrix[j, i] = min_GED


    print(f"Calculating the cost matrix: {t.time() - basetime}s")
    print("\n")
    return cost_matrix