import networkx as nx
import time as t
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from functools import lru_cache
import concurrent.futures

def encode_graph(graph):
    node_labels = "".join(sorted([f"{n}:{graph.nodes[n]['label']}" for n in graph.nodes]))
    edge_labels = "".join(sorted([f"{u}-{v}:{graph.edges[u, v]['label']}" for u, v in graph.edges]))
    return hash(node_labels + edge_labels)

def build_nt(graph, root, height, k):
    tree = nx.DiGraph()
    tree.add_node(root, label=graph.nodes[root]["label"], height=0)
    D = {}
    D[root] = 0
    Phi = {}
    Phi[root] = root
    for i in range(1, height):
        F = {}
        for v in sorted(tree.nodes):
            if tree.out_degree(v) == 0:
                for u in graph.neighbors(Phi[v]):
                    if u not in D:
                        D[u] = i
                    if D[u] + k >= i:
                        if u not in F:
                            c = u
                            tree.add_node(c, label=graph.nodes[c]["label"], height=i)
                            Phi[c] = u
                            F[u] = c
                        tree.add_edge(v, F[u], label=graph.edges[v, F[u]]["label"])

    tree.graph["encoding"] = encode_graph(tree)

    return tree


def sdted(treee1, treee2, subgraph_dict1, subgraph_dict2, cache):
    def pad(tree, number):
        root = next(iter(tree.nodes))
        
        current_degree = tree.degree[root]
        if current_degree >= number:
            return tree

        missing_children = number - current_degree
        
        new_node = max(tree.nodes) + 1
        
        # Knoten und Kanten effizient hinzufügen
        new_nodes = [(new_node + i, {"label": "pad", "height": tree.nodes[root]["height"] + 1}) for i in range(missing_children)]
        new_edges = [(root, new_node + i) for i in range(missing_children)]
        
        tree.add_nodes_from(new_nodes)
        tree.add_edges_from(new_edges)

        return tree

    @lru_cache(maxsize=None)
    def recusive_sdted(tree1, tree2, depth):
        key = (tree1.graph["encoding"], tree2.graph["encoding"])
        if key in cache:
            return cache[key]

        # n is the maximum number of children of the root of the two trees
        n = max(tree1.degree[next(iter(tree1.nodes))], tree2.degree[next(iter(tree2.nodes))])

        # add undefined nodes to the trees

        tree1_padded = tree1
        tree2_padded = tree2

        if tree1.degree[next(iter(tree1.nodes))] != tree2.degree[next(iter(tree2.nodes))]:

            tree1_padded = pad(tree1, n)
            tree2_padded = pad(tree2, n)

        # create the cost matrix as n x n matrix
        cost_matrix = np.zeros((n, n))
        # fill the cost matrix

        root1 = next(iter(tree1_padded.nodes))
        root2 = next(iter(tree2_padded.nodes))

        children1 = sorted(tree1_padded.neighbors(next(iter(tree1_padded.nodes))))
        children2 = sorted(tree2_padded.neighbors(next(iter(tree2_padded.nodes))))

        for i in range(n):
            for j in range(n):

                child1_ident = children1[i]
                child2_ident = children2[j]

                child1 = tree1_padded.nodes[child1_ident]
                child2 = tree2_padded.nodes[child2_ident]

                child1_label = child1["label"]
                child2_label = child2["label"]

                if child1_label != "pad" or child2_label != "pad":
                    if child1_label != "pad" and child2_label == "pad":
                        cost_matrix[i][j] = (tree1_padded.nodes[child1_ident]["cost"] + 1) * (1/(1+depth+1))
                    elif child2_label != "pad" and child1_label == "pad":
                        cost_matrix[i][j] = (tree2_padded.nodes[child2_ident]["cost"] + 1) * (1/(1+depth+1))
                    else:
                        temp_cost = 1 if hash(tree1_padded.edges[root1, child1_ident]["label"]) != hash(tree2_padded.edges[root2, child2_ident]["label"]) else 0
                        
                        recursive_cost = recusive_sdted(subgraph_dict1[child1_ident], subgraph_dict2[child2_ident], depth + 1)
                        cost_matrix[i][j] = (recursive_cost + temp_cost) 


        # calculate the cost of the roots
        cost_root = 1 
        if tree1_padded.nodes[root1]["label"] == tree2_padded.nodes[root2]["label"]:
            cost_root = 0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        cost = 0
        for i in range(n): 
            cost += cost_matrix[row_ind[i]][col_ind[i]]



        result = (cost + cost_root) * (1/(1+depth))

        cache[key] = result

        return result

    result = recusive_sdted(treee1, treee2, 1)
    cache[(treee1.graph["encoding"], treee2.graph["encoding"])] = result
    return result


def calculate_costs(tree):
    for node in sorted(tree.nodes):
        cost = 1
        queue = deque([node])
        visited = set([node])

        while queue:
            current_node = queue.popleft()
            for neighbor in tree.neighbors(current_node):
                if neighbor not in visited and tree.nodes[neighbor]["height"] > tree.nodes[current_node]["height"]:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    cost += 2
        tree.nodes[node]["cost"] = cost

    return tree


def create_subgraph(graph, node):
    subgraph = nx.DiGraph()
    subgraph.add_node(node, label=graph.nodes[node]["label"], height=graph.nodes[node]["height"], cost=graph.nodes[node]["cost"])

    current_level = [node]
    next_level = []

    while current_level:
        for current_node in current_level:
            for neighbor in graph.neighbors(current_node):
                if graph.nodes[neighbor]["height"] > graph.nodes[current_node]["height"]:
                    if neighbor not in subgraph.nodes:
                        subgraph.add_node(neighbor, label=graph.nodes[neighbor]["label"], height=graph.nodes[neighbor]["height"], cost=graph.nodes[neighbor]["cost"])
                        subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])
                        next_level.append(neighbor)
                    else:
                        subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])
                    

        current_level = next_level
        next_level = []

    subgraph.graph["encoding"] = encode_graph(subgraph)

    return subgraph


def create_subgraph_dict(graph):
    subgraph_dict = {}
    for node in sorted(graph.nodes):
        subgraph_dict[node] = create_subgraph(graph, node)
    return subgraph_dict


def create_nt_dict(graphs, height, k):
    nt_dict = {}
    for graph_id in graphs:
        for node in sorted(graphs[graph_id].nodes):
            nt = build_nt(graphs[graph_id], node, height, k)
            nt_dict[(graph_id, node)] = calculate_costs(nt), create_subgraph_dict(nt)
    return nt_dict

from itertools import zip_longest

# def derive_edit_path(graph1, graph2, row_ind, col_ind):
#     edit_path = []
#     matched_nodes1 = {list(graph1.nodes)[i] for i in row_ind}
#     matched_nodes2 = {list(graph2.nodes)[j] for j in col_ind}

#     nodes_missing_edges = set()

#     visited_edges = set()

#     for i, j in zip(row_ind, col_ind): 
#         node1 = list(graph1.nodes)[i]
#         node2 = list(graph2.nodes)[j]

#         # Falls Labels unterschiedlich sind → Relabeling
#         if graph1.nodes[node1]["label"] != graph2.nodes[node2]["label"]:
#             edit_path.append(f"relabel {node1} -> {node2}")

#         # schaue nur nachbarn an, die ebenfalls gematcht wurden
#         neighbors1 = set(graph1.neighbors(node1)) & matched_nodes1
#         neighbors2 = set(graph2.neighbors(node2)) & matched_nodes2

#         corresponding_neighbors = list(zip_longest(neighbors1, neighbors2))

#         for neighbor1, neighbor2 in corresponding_neighbors:
#             if (node1, neighbor1) in visited_edges or (neighbor1, node1) in visited_edges:
#                 continue
#             if (node2, neighbor2) in visited_edges or (neighbor2, node2) in visited_edges:
#                 continue

#             if (node1, neighbor1) in graph1.edges and (node2, neighbor2) in graph2.edges:
#                 if graph1.edges[node1, neighbor1]["label"] != graph2.edges[node2, neighbor2]["label"]:
#                     edit_path.append(f"relabel edge {node1} -> {neighbor1} -> {node2} -> {neighbor2}")

#             if(node2, neighbor2) in graph2.edges and (node1, neighbor1) not in graph1.edges:
#                 nodes_missing_edges.add(node1)
#             if(node1, neighbor1) in graph1.edges and (node2, neighbor2) not in graph2.edges:
#                 edit_path.append(f"delete edge: {node1} -> {neighbor1}")

#             visited_edges.add((node1, neighbor1))
#             visited_edges.add((neighbor1, node1))
#             visited_edges.add((node2, neighbor2))
#             visited_edges.add((neighbor2, node2))

#     for node1 in nodes_missing_edges: #TODO
#         for node2 in nodes_missing_edges:
#             if node1 != node2:
#                 # get the corresponding nodes in graph2
#                 if node1 in graph1.nodes and node2 in graph1.nodes:
#                     index1 = list(graph1.nodes).index(node1)
#                     index2 = list(graph1.nodes).index(node2)
#                     if index1 <= len(col_ind) and index2 <= len(col_ind):
#                         corresponding_node1 = list(graph2.nodes)[col_ind[index1]]
#                         corresponding_node2 = list(graph2.nodes)[col_ind[index2]]
#                         if (corresponding_node1, corresponding_node2) in graph2.edges:
#                             if (node1, node2) not in visited_edges or (node2, node1) not in visited_edges:
#                                 edit_path.append(f"insert edge: {node1} -> {node2}")
#                                 visited_edges.add((node1, node2))
#                                 visited_edges.add((node2, node1))

#     # Knoten, die nicht gematcht wurden, müssen gelöscht oder eingefügt werden #TODO: Check if this is correct
#     unmatched_nodes1 = set(graph1.nodes) - matched_nodes1
#     unmatched_nodes2 = set(graph2.nodes) - matched_nodes2

#     for node in unmatched_nodes1:
#         edit_path.append(f"delete node: {node}")
#         for neighbor in graph1.neighbors(node):
#             if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
#                 edit_path.append(f"delete edge: {node} -> {neighbor}")
#                 visited_edges.add((node, neighbor))
#                 visited_edges.add((neighbor, node))

#     for node in unmatched_nodes2:
#         edit_path.append(f"insert node: {node}")
#         for neighbor in graph2.neighbors(node):
#             if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
#                 edit_path.append(f"insert edge: {node} -> {neighbor}")
#                 visited_edges.add((node, neighbor))
#                 visited_edges.add((neighbor, node))

#     return edit_path

def derive_edit_path(graph1, graph2, row_ind, col_ind):
    edit_path = []
    nodes1 = sorted(graph1.nodes)
    nodes2 = sorted(graph2.nodes)

    visited_edges = set()
    
    # Erzeuge das Matching
    mapping = {}
    mapping_inv = {}
    matched_nodes1 = set()
    matched_nodes2 = set()
    for i, j in zip(row_ind, col_ind):
        node1 = nodes1[i]
        node2 = nodes2[j]
        mapping[node1] = node2
        mapping_inv[node2] = node1
        matched_nodes1.add(node1)
        matched_nodes2.add(node2)
        if graph1.nodes[node1]["label"] != graph2.nodes[node2]["label"]:
            edit_path.append(f"relabel node {node1} -> {node2}")
    
    # Unmatched Knoten in Graph1: löschen
    unmatched_nodes1 = set(graph1.nodes) - matched_nodes1
    for node in unmatched_nodes1:
        edit_path.append(f"delete node: {node}")
        for neighbor in graph1.neighbors(node):
            if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
                visited_edges.add((node, neighbor))
                visited_edges.add((neighbor, node))
                edit_path.append(f"delete edge: {node}-{neighbor}")
    
    # Unmatched Knoten in Graph2: einfügen
    unmatched_nodes2 = set(graph2.nodes) - matched_nodes2
    for node in unmatched_nodes2:
        edit_path.append(f"insert node: {node}")
        for neighbor in graph2.neighbors(node):
            if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
                visited_edges.add((node, neighbor))
                visited_edges.add((neighbor, node))
                edit_path.append(f"insert edge: {node}-{neighbor}")

    # TODO Hier fehler mit der 8 lösen

    for (u,v) in graph1.edges:
        if u not in mapping or v not in mapping:
            continue
        if (u,v) in visited_edges or (v,u) in visited_edges:
            continue
        if (mapping[u], mapping[v]) not in graph2.edges:
            edit_path.append(f"delete edge: {u}-{v}")
            visited_edges.add((u,v))
        elif graph1.edges[u,v]["label"] != graph2.edges[mapping[u], mapping[v]]["label"]:
            edit_path.append(f"relabel edge: {u}-{v} -> {mapping[u]}-{mapping[v]}")
            visited_edges.add((u,v))


    for (u,v) in graph2.edges:
        if u not in mapping_inv or v not in mapping_inv:
            continue
        if (u,v) in visited_edges or (v,u) in visited_edges:
            continue
        if (mapping_inv[u], mapping_inv[v]) not in graph1.edges:
            edit_path.append(f"insert edge: {mapping_inv[u]}-{mapping_inv[v]}")
            visited_edges.add((u,v))
        elif graph1.edges[mapping_inv[u], mapping_inv[v]]["label"] != graph2.edges[u,v]["label"]:
            edit_path.append(f"relabel edge: {u}-{v} -> {mapping_inv[u]}-{mapping_inv[v]}")
            visited_edges.add((u,v))
    
    return edit_path


def calculate_GED_bgm(graph1, graph2, nt_dict, cache):
    n1, n2 = len(graph1.nodes), len(graph2.nodes)
    cost_matrix = np.full((n1, n2), np.inf)  # Initialisiere mit hohen Kosten

    # nodes1 = list(graph1.nodes)
    # nodes2 = list(graph2.nodes)

    nodes1 = sorted(graph1.nodes)
    nodes2 = sorted(graph2.nodes)

    max_value = 0

    for i, node1 in enumerate(nodes1):
        for j, node2 in enumerate(nodes2):
            nt1 = nt_dict[(graph1.graph["id"], node1)][0]
            nt2 = nt_dict[(graph2.graph["id"], node2)][0]
            nt1_subgraph = nt_dict[(graph1.graph["id"], node1)][1]
            nt2_subgraph = nt_dict[(graph2.graph["id"], node2)][1]

            # Grundkosten aus SDTED
            current_result = sdted(nt1, nt2, nt1_subgraph, nt2_subgraph, cache)
            cost_matrix[i, j] = current_result

            if current_result > max_value:
                max_value = current_result

    # Optimiere das Matching mit dem Hungarian Algorithmus
    # normalize the cost matrix
    cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
    cost_matrix += np.random.uniform(0, 0.0001, cost_matrix.shape)  # Kleine zufällige Störungen hinzufügen

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Berechne den finalen Edit-Path mit den Edge Costs
    edit_path = derive_edit_path(graph1, graph2, row_ind, col_ind)
    min_GED = len(edit_path)

    return row_ind, col_ind, min_GED, edit_path


def calculate_cost_matrix(graphs, height=8, k=0):
    basetime = t.time()
    cache = {}
    nt_dict = create_nt_dict(graphs, height, k)

    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0)
    edit_paths = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(calculate_GED_bgm, graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict, cache)

        for (i, j), future in futures.items():
            row_ind, col_ind, min_GED, edit_path = future.result()
            cost_matrix[i, j] = cost_matrix[j, i] = min_GED
            edit_paths[(i, j)] = edit_paths[(j, i)] = edit_path

    print(f"Total time: {t.time() - basetime}")
    return cost_matrix, edit_paths





# OLD CODE WHICH USED SDTED AS GED CALCULATION

# def calculate_GED(graph1, graph2, nt_dict):
#     min_GED = float("inf")
#     for node1 in graph1.nodes:
#         for node2 in graph2.nodes:
#             nt1 = nt_dict[(graph1.graph["id"], node1)][0]
#             nt2 = nt_dict[(graph2.graph["id"], node2)][0]
#             # diff_nodes = abs(len(nt1.nodes) - len(nt2.nodes))
#             # diff_edges = abs(len(nt1.edges) - len(nt2.edges))
#             # if diff_nodes/2 >= min_GED:
#             #     continue
#             # if diff_edges >= min_GED:
#             #     continue
#             # else:
#             GED = sdted(nt1, nt2, nt_dict[(graph1.graph["id"], node1)][1],nt_dict[(graph2.graph["id"], node2)][1]) #TODO: Add subgraphs here
#             print(graph1.graph["id"],graph2.graph["id"], node1, node2 ,GED)

#             if GED < min_GED:
#                 min_GED = GED

#             # print("Time to compute one GED for NT-Root Nodes " + str(node1) +" and " + str(node2) +": " + str(t.time() - basetime))

#     return min_GED

# def calculate_cost_matrix(graphs):
#     basetime = t.time()
#     nt_dict = create_nt_dict(graphs, 8, 0)

#     graph_ids = list(graphs.keys())
#     cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)

#     for i, j in combinations(range(len(graph_ids)), 2):
#         cost_matrix[i, j] = cost_matrix[j, i] = calculate_GED(graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict)

#     print(f"Total time: {t.time() - basetime}")
#     return np.matrix(cost_matrix)


# def calculate_GED_bgm(graph1, graph2, nt_dict, cache):
#     n1, n2 = len(graph1.nodes), len(graph2.nodes)
#     cost_matrix = np.full((n1, n2), np.inf)  # Verwende eine hohe Default-Kosten für nicht existierende Knoten

#     nodes1 = list(graph1.nodes)
#     nodes2 = list(graph2.nodes)

#     # Berechne SDTED für alle möglichen Knotenpaare
#     for i, node1 in enumerate(nodes1):
#         for j, node2 in enumerate(nodes2):
#             nt1 = nt_dict[(graph1.graph["id"], node1)][0]
#             nt2 = nt_dict[(graph2.graph["id"], node2)][0]
#             nt1_subgraph = nt_dict[(graph1.graph["id"], node1)][1]
#             nt2_subgraph = nt_dict[(graph2.graph["id"], node2)][1]

#             cost_matrix[i, j] = sdted(nt1, nt2, nt1_subgraph, nt2_subgraph, cache)[0]  # Speichere SDTED-Werte

#     # Optimales Matching mit dem Hungarian Algorithmus
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     # Berechne GED basierend auf dem Edit-Path
#     edit_path = derive_edit_path(graph1, graph2, row_ind, col_ind)
#     min_GED = len(edit_path)  # GED entspricht der Anzahl der benötigten Edit-Operationen

#     return row_ind, col_ind, min_GED


# def calculate_GED_parallel(graph1, graph2, nt_dict, cache):
#     min_GED = float("inf")
#     min_edit_path = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for node1 in graph1.nodes:
#             for node2 in graph2.nodes:
#                 nt1 = nt_dict[(graph1.graph["id"], node1)][0]
#                 nt2 = nt_dict[(graph2.graph["id"], node2)][0]
#                 nt1_subgraph = nt_dict[(graph1.graph["id"], node1)][1]
#                 nt2_subgraph = nt_dict[(graph2.graph["id"], node2)][1]

#                 diff_nodes = abs(len(nt1.nodes) - len(nt2.nodes))
#                 diff_edges = abs(len(nt1.edges) - len(nt2.edges))

#                 if diff_nodes/2 >= min_GED:
#                     continue
#                 if diff_edges >= min_GED:
#                     continue

#                 futures.append(executor.submit(sdted, nt1, nt2, nt1_subgraph, nt2_subgraph, cache))

#         for future in concurrent.futures.as_completed(futures):
#             GED = future.result()[0]
#             if GED < min_GED:
#                 min_GED = GED
#                 min_edit_path = future.result()[1]

#     return (min_GED, min_edit_path)

# def calculate_cost_matrix(graphs, height=8, k=0):
#     basetime = t.time()
#     cache = {}
#     nt_dict = create_nt_dict(graphs, height, k)

#     graph_ids = list(graphs.keys())
#     cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)
#     edit_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = {}
#         for i, j in combinations(range(len(graph_ids)), 2):
#             futures[(i, j)] = executor.submit(calculate_GED_parallel, graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict, cache)

#         for (i, j), future in futures.items():
#             cost_matrix[i, j] = cost_matrix[j, i] = future.result()[0]
#             edit_matrix[i, j] = edit_matrix[j, i] = future.result()[1]

#     print(f"Total time: {t.time() - basetime}")
#     return np.matrix(cost_matrix)