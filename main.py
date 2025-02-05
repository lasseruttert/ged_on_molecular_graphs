import networkx as nx
import time as t
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from functools import lru_cache
import matplotlib.pyplot as plt
import concurrent.futures

# ? These functions are not of any use in our implementation, but it is possible to use them to calculate the costs of the different operations in the GED calculation
# ? For example, the cost of relabeling a edge could be different based on the label of the edge and its nodes, this is especially interesting in molecular graphs, as you can consider the energy of the bond between the atoms as the cost of relabeling the edge
# ? The places, where these functions should be used are marked with a comment in the code, but not checked for correctness

def cost_insert_node(node_label):
    return 1

def cost_delete_node(node_label):
    return 1

def cost_insert_edge(edge_label):
    return 1

def cost_delete_edge(edge_label):
    return 1

def cost_relabel_node(node_label1, node_label2):
    return 1

def cost_relabel_edge(edge_label1, edge_label2):
    return 1

# ? The following function is our implementation of canonical encoding of a tree, which is used to encode the neighborhood trees in the SDTED calculation
# ? As seen below, a different version of the encoding sacrifices some structural information, but is much faster to compute

def encode_graph(graph):
    """
    * encodes a graph via canonical encoding and hashing

    * param graph: a networkx graph

    * return: a hash of the canonical encoding

    * description:
    * The function encodes a graph via a canonical encoding, which is a string representation of the graph
    * The string has the following format: (label(children))
    * A canonical encoding is built by recursively encoding the children of a node
    * Two canonical encodings are only equal if the graphs are isomorphic
    * The canonical encoding is then hashed to a unique hash value
    """
    def canonical_encoding(graph, node = None):
        if node is None:
            node = next(iter(graph))  # Starte mit einem beliebigen Knoten
        # Rekursiv die Kinderknoten kodieren
        children = sorted([canonical_encoding(graph, child) for child in graph.neighbors(node) if graph.nodes[child]["height"] > graph.nodes[node]["height"]])
        # Knotenlabel und kodierte Kinder kombinieren
        return (f"({graph.nodes[node]['label']}" + "".join(children) + ")")
    
    return hash(canonical_encoding(graph))


# def encode_graph(graph):
    # node_labels = "".join(sorted([f"{graph.nodes[n]['label']}" for n in graph.nodes]))
    # edge_labels = "".join(sorted([f"{graph.nodes[u]['label']}-{graph.nodes[v]['label']}:{graph.edges[u, v]['label']}" for u, v in graph.edges]))
    # return hash(node_labels + edge_labels)


# ? The following functions build_nt and sdted were implemented based on the given pseudocode in the paper "Approximating the Graph Edit Distance with Compact Neighborhood Representations"

def build_nt(graph, root, height, k):
    """
    * builds a neighborhood tree of a graph with a given root node, height and k

    * param graph: a networkx graph
    * param root: a node in the graph to be the root of the neighborhood tree
    * param height: the height of the neighborhood tree
    * param k: the maximum height difference for redundancy elimination

    * return: a networkx DiGraph object representing the neighborhood tree
    
    * description:
    * The function builds a neighborhood tree of a graph with a given root node, height and k
    TODO add more description
    """
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

        # children1 = sorted(tree1_padded.neighbors(next(iter(tree1_padded.nodes))))
        # children2 = sorted(tree2_padded.neighbors(next(iter(tree2_padded.nodes))))

        nodes_list = list(tree1_padded.nodes)  # Sicherstellen, dass wir über eine feste Liste iterieren
        children1 = sorted(tree1_padded.neighbors(nodes_list[0]))  # Nimm das erste Element der Liste
        nodes_list = list(tree2_padded.nodes)  # Sicherstellen, dass wir über eine feste Liste iterieren
        children2 = sorted(tree2_padded.neighbors(nodes_list[0]))  # Nimm das erste Element der Liste

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
                        cost_matrix[i][j] = (tree1_padded.nodes[child1_ident]["cost"] + 1) #* (1/(1+depth+1))
                        # cost_matrix[i][j] = (tree1_padded.nodes[child1_ident]["cost"] + cost_insert_edge(tree1_padded.edges[root1, child1_ident]["label"])) * (1/(1+depth+1))
                    elif child2_label != "pad" and child1_label == "pad":
                        cost_matrix[i][j] = (tree2_padded.nodes[child2_ident]["cost"] + 1) #* (1/(1+depth+1))
                        # cost_matrix[i][j] = (tree2_padded.nodes[child2_ident]["cost"] + cost_insert_edge(tree2_padded.edges[root2, child2_ident]["label"])) * (1/(1+depth+1))
                    else:
                        temp_cost = 1 if hash(tree1_padded.edges[root1, child1_ident]["label"]) != hash(tree2_padded.edges[root2, child2_ident]["label"]) else 0
                        # temp_cost = cost_relabel_edge(tree1_padded.edges[root1, child1_ident]["label"], tree2_padded.edges[root2, child2_ident]["label"]) if tree1_padded.edges[root1, child1_ident]["label"] != tree2_padded.edges[root2, child2_ident]["label"] else 0

                        # check cache
                        if (subgraph_dict1[child1_ident].graph["encoding"], subgraph_dict2[child2_ident].graph["encoding"]) in cache:
                            recursive_cost = cache[(subgraph_dict1[child1_ident].graph["encoding"], subgraph_dict2[child2_ident].graph["encoding"])]
                        else:
                            recursive_cost = recusive_sdted(subgraph_dict1[child1_ident], subgraph_dict2[child2_ident], depth + 1)
                        cost_matrix[i][j] = (recursive_cost + temp_cost) 

        # calculate the cost of the roots
        cost_root = 0 
        if tree1_padded.nodes[root1]["label"] != tree2_padded.nodes[root2]["label"]:
            cost_root = 1
            # cost_root = cost_relabel_node(tree1_padded.nodes[root1]["label"], tree2_padded.nodes[root2]["label"])

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

# ? The following functions are used to :
# ? 1. frontload the costs of insertion and deletion of nodes in the neighborhood trees
# ? 2. create a subgraph of the neighborhood tree, starting from a specific node
# ? 3. create a dictionary of subgraphs for each node in the neighborhood tree
# ? 4. create a dictionary of neighborhood trees for each node in the graph
# ? while this takes a lot of time upfront, it makes the calculations of the SDTED much faster

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
                    cost += 2 * (1/(1+tree.nodes[neighbor]["height"]+1))
                    #cost += cost_insert_edge(tree.edges[current_node, neighbor]["label"]) + cost_insert_node(tree.nodes[neighbor]["label"])
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

# ? The following functions are used to calculate the cost matrix, edit paths and matchings between the graphs

def derive_edit_path(graph1, graph2, row_ind, col_ind):
    edit_path = []
    # edit_cost = 0
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
            edit_path.append(f"relabel node: {node1} -> {node2}")
            # edit_cost += cost_relabel_node(graph1.nodes[node1]["label"], graph2.nodes[node2]["label"])
    
    # Unmatched Knoten in Graph1: löschen
    unmatched_nodes1 = set(graph1.nodes) - matched_nodes1
    for node in unmatched_nodes1:
        for neighbor in graph1.neighbors(node):
            if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
                visited_edges.add((node, neighbor))
                visited_edges.add((neighbor, node))
                edit_path.append(f"delete edge: {node}-{neighbor}")
                # edit_cost += cost_delete_edge(graph1.edges[node, neighbor]["label"])
        edit_path.append(f"delete node: {node}")
        # edit_cost += cost_delete_node(graph1.nodes[node]["label"])
    
    # Unmatched Knoten in Graph2: einfügen
    unmatched_nodes2 = set(graph2.nodes) - matched_nodes2
    for node in unmatched_nodes2:
        edit_path.append(f"insert node: {node}")
        # edit_cost += cost_insert_node(graph2.nodes[node]["label"])

    for node in unmatched_nodes2:
        for neighbor in graph2.neighbors(node):
                if neighbor not in mapping_inv:
                    if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
                        visited_edges.add((node, neighbor))
                        visited_edges.add((neighbor, node))
                        edit_path.append(f"insert n_edge: {node}-{neighbor}")
                        # edit_cost += cost_insert_edge(graph2.edges[node, neighbor]["label"])
                elif neighbor in mapping_inv:
                    if (node, neighbor) not in visited_edges or (neighbor, node) not in visited_edges:
                        visited_edges.add((node, neighbor))
                        visited_edges.add((neighbor, node))
                        edit_path.append(f"insert h_edge: {node}-{mapping_inv[neighbor]}")
                        # edit_cost += cost_insert_edge(graph2.edges[node, mapping_inv[neighbor]]["label"])

    for (u,v) in graph1.edges:
        if u not in mapping or v not in mapping:
            continue
        if (u,v) in visited_edges or (v,u) in visited_edges:
            continue
        if (mapping[u], mapping[v]) not in graph2.edges:
            edit_path.append(f"delete edge: {u}-{v}")
            # edit_cost += cost_delete_edge(graph1.edges[u,v]["label"])
            visited_edges.add((u,v))
        elif graph1.edges[u,v]["label"] != graph2.edges[mapping[u], mapping[v]]["label"]:
            edit_path.append(f"relabel edge: {u}-{v} -> {mapping[u]}-{mapping[v]}")
            # edit_cost += cost_relabel_edge(graph1.edges[u,v]["label"], graph2.edges[mapping[u], mapping[v]]["label"])
            visited_edges.add((u,v))


    for (u,v) in graph2.edges:
        if u not in mapping_inv or v not in mapping_inv:
            continue
        if (u,v) in visited_edges or (v,u) in visited_edges:
            continue
        if (mapping_inv[u], mapping_inv[v]) not in graph1.edges:
            edit_path.append(f"insert edge: {mapping_inv[u]}-{mapping_inv[v]}")
            # edit_cost += cost_insert_edge(graph2.edges[u,v]["label"])
            visited_edges.add((u,v))
        elif graph1.edges[mapping_inv[u], mapping_inv[v]]["label"] != graph2.edges[u,v]["label"]:
            edit_path.append(f"relabel r_edge: {u}-{v} -> {mapping_inv[u]}-{mapping_inv[v]}")
            # edit_cost += cost_relabel_edge(graph1.edges[mapping_inv[u], mapping_inv[v]]["label"], graph2.edges[u,v]["label"])
            visited_edges.add((u,v))
    
    return edit_path #, edit_cost


def calculate_GED_bgm(graph1, graph2, nt_dict = None, cache = {}):
    if nt_dict is None:
        nt_dict = create_nt_dict({graph1.graph["id"]: graph1, graph2.graph["id"]: graph2}, 8, 0)
    
    n1, n2 = len(graph1.nodes), len(graph2.nodes)
    cost_matrix = np.full((n1, n2), np.inf)  # Initialisiere mit hohen Kosten

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

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # get a list of matched nodes
    matching = [(nodes1[i], nodes2[j]) for i, j in zip(row_ind, col_ind)]

    # Berechne den finalen Edit-Path mit den Edge Costs
    edit_path = derive_edit_path(graph1, graph2, row_ind, col_ind)
    min_GED = len(edit_path)

    return row_ind, col_ind, min_GED, edit_path, matching


def calculate_cost_matrix(graphs, height=8, k=0):
    basetime = t.time()
    cache = {}
    nt_dict = create_nt_dict(graphs, height, k)
    print(f"Create the NT dictionary: {t.time() - basetime}s")
    basetime = t.time()

    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0)
    edit_paths = {}
    matchings = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(calculate_GED_bgm, graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict, cache)

        for (i, j), future in futures.items():
            row_ind, col_ind, min_GED, edit_path, matching = future.result()
            cost_matrix[i, j] = cost_matrix[j, i] = min_GED
            edit_paths[(i, j)] = edit_paths[(j, i)] = edit_path
            matchings[(i, j)] = matchings[(j, i)] = matching

    print(f"Calculating the cost matrix: {t.time() - basetime}s")
    print("\n")
    return cost_matrix, edit_paths, matchings


# ? The following functions are used to check if the edit path is valid and to apply the edit path to the graph, to check if the graphs are isomorphic and to print the two graphs

def graph_matcher(graph1, graph2, edit_path, matching):
    # Erzeuge Kopien der Graphen
    graph1 = graph1.copy()

    for action in edit_path:
        if action.startswith("relabel node"):
            _, nodes = action.split(":")
            node1, node2 = nodes.split("->")
            graph1.nodes[int(node1)]["label"] = graph2.nodes[int(node2)]["label"]
        
        elif action.startswith("delete node"):
            _, node = action.split(":")
            graph1.remove_node(int(node))
        
        elif action.startswith("insert node"):
            _, node1 = action.split(":")
            # find the node in graph2
            graph1.add_node(int(node1), label=graph2.nodes[int(node1)]["label"])
        
        elif action.startswith("relabel edge"):
            _, edges = action.split(":")
            edge1, edge2 = edges.split("->")
            u1, v1 = map(int, edge1.split("-"))
            u2, v2 = map(int, edge2.split("-"))
            graph1.edges[u1, v1]["label"] = graph2.edges[u2, v2]["label"]
        
        elif action.startswith("relabel r_edge"):
            _, edges = action.split(":")
            edge2, edge1 = edges.split("->")
            u1, v1 = map(int, edge1.split("-"))
            u2, v2 = map(int, edge2.split("-"))
            graph1.edges[u1, v1]["label"] = graph2.edges[u2, v2]["label"]
        
        elif action.startswith("delete edge"):
            _, edge = action.split(":")
            u, v = map(int, edge.split("-"))
            graph1.remove_edge(u, v)
        
        elif action.startswith("insert edge"):
            _, edge = action.split(":")
            u, v = map(int, edge.split("-"))
            u_matching = None
            v_matching = None
            for tupel in matching:
                if tupel[0] == u:
                    u_matching = tupel[1]
                if tupel[0] == v:
                    v_matching = tupel[1]
            graph1.add_edge(u, v, label=graph2.edges[u_matching, v_matching]["label"])
        
        elif action.startswith("insert n_edge"):
            _, edge = action.split(":")
            u, v = map(int, edge.split("-"))
            graph1.add_edge(u, v, label=graph2.edges[u, v]["label"])
        
        elif action.startswith("insert h_edge"):
            _, edge = action.split(":")
            u, v = map(int, edge.split("-"))
            v_matching = None
            for tupel in matching:
                if tupel[0] == v:
                    v_matching = tupel[1]
            graph1.add_edge(u, v, label=graph2.edges[u, v_matching]["label"])

    return graph1

def isomorph_check(graph1, graph2):
    # Überprüfe, ob die Anzahl der Knoten und Kanten gleich ist
    if len(graph1.nodes) != len(graph2.nodes) or len(graph1.edges) != len(graph2.edges):
        return False

    # Verwende die eingebaute Funktion von NetworkX, um die Isomorphie zu überprüfen
    return nx.is_isomorphic(graph1, graph2, node_match=node_match, edge_match=edge_match)

def node_match(n1, n2):
    # Überprüfe, ob die Labels der Knoten übereinstimmen
    return n1['label'] == n2['label']

def edge_match(e1, e2):
    # Überprüfe, ob die Labels der Kanten übereinstimmen
    return e1['label'] == e2['label']


def print_two_graphs(graph1, graph2, layout='spring'):

    graph1_label = graph1.graph['id']
    graph2_label = graph2.graph['id']

    fig, axes = plt.subplots(1, 2, figsize=(30, 15))

    if layout == 'spring':
        pos1 = nx.spring_layout(graph1)
        pos2 = nx.spring_layout(graph2)
    elif layout == 'circular':
        pos1 = nx.circular_layout(graph1)
        pos2 = nx.circular_layout(graph2)
    elif layout == 'kamada_kawai':
        pos1 = nx.kamada_kawai_layout(graph1)
        pos2 = nx.kamada_kawai_layout(graph2)
    else:
        raise ValueError("Unsupported layout type. Use 'spring', 'circular', or 'kamada_kawai'.")

    # Plot graph1
    node_labels1 = nx.get_node_attributes(graph1, 'label')
    edge_labels1 = nx.get_edge_attributes(graph1, 'label')
    nx.draw(graph1, pos1, with_labels=True, labels=node_labels1, ax=axes[0])
    nx.draw_networkx_edge_labels(graph1, pos1, edge_labels=edge_labels1, ax=axes[0])
    axes[0].set_title("Graph " + str(graph1_label))

    # Plot graph2
    node_labels2 = nx.get_node_attributes(graph2, 'label')
    edge_labels2 = nx.get_edge_attributes(graph2, 'label')
    nx.draw(graph2, pos2, with_labels=True, labels=node_labels2, ax=axes[1])
    nx.draw_networkx_edge_labels(graph2, pos2, edge_labels=edge_labels2, ax=axes[1])
    axes[1].set_title("Graph " + str(graph2_label))

    plt.show()

    return None






# ! OLD CODE WHICH USED SDTED AS GED CALCULATION (IGNORE THIS)

# // def calculate_GED_parallel(graph1, graph2, nt_dict, cache):
# //     min_GED = float("inf")
# //     min_edit_path = []
# //     with concurrent.futures.ThreadPoolExecutor() as executor:
# //         futures = []
# //         for node1 in graph1.nodes:
# //             for node2 in graph2.nodes:
# //                 nt1 = nt_dict[(graph1.graph["id"], node1)][0]
# //                 nt2 = nt_dict[(graph2.graph["id"], node2)][0]
# //                 nt1_subgraph = nt_dict[(graph1.graph["id"], node1)][1]
# //                 nt2_subgraph = nt_dict[(graph2.graph["id"], node2)][1]

# //                 diff_nodes = abs(len(nt1.nodes) - len(nt2.nodes))
# //                 diff_edges = abs(len(nt1.edges) - len(nt2.edges))

# //                 if diff_nodes/2 >= min_GED:
# //                     continue
# //                 if diff_edges >= min_GED:
# //                     continue

# //                 futures.append(executor.submit(sdted, nt1, nt2, nt1_subgraph, nt2_subgraph, cache))

# //         for future in concurrent.futures.as_completed(futures):
# //             GED = future.result()[0]
# //             if GED < min_GED:
# //                 min_GED = GED
# //                 min_edit_path = future.result()[1]

# //     return (min_GED, min_edit_path)

# // def calculate_cost_matrix(graphs, height=8, k=0):
# //     basetime = t.time()
# //     cache = {}
# //     nt_dict = create_nt_dict(graphs, height, k)

# //     graph_ids = list(graphs.keys())
# //     cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)
# //     edit_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)

# //     with concurrent.futures.ThreadPoolExecutor() as executor:
# //         futures = {}
# //         for i, j in combinations(range(len(graph_ids)), 2):
# //             futures[(i, j)] = executor.submit(calculate_GED_parallel, graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict, cache)

# //         for (i, j), future in futures.items():
# //             cost_matrix[i, j] = cost_matrix[j, i] = future.result()[0]
# //             edit_matrix[i, j] = edit_matrix[j, i] = future.result()[1]

# //     print(f"Total time: {t.time() - basetime}")
# //     return np.matrix(cost_matrix)