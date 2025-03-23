import os
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from collections import deque
from scipy.optimize import linear_sum_assignment
import concurrent.futures
from functools import lru_cache

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
        # start with the first node if no node is given
        if node is None:
            node = next(iter(graph)) 
        # encode the children of the node recursively
        children = sorted([canonical_encoding(graph, child) for child in graph.neighbors(node) if graph.nodes[child]["height"] > graph.nodes[node]["height"]])
        # reconstruct the canonical encoding of the node and its children
        return (f"({graph.nodes[node]['label']}" + "".join(children) + ")")
    
    return hash(canonical_encoding(graph))


# ? The following functions build_nt and sdted were implemented based on the given pseudocode in the paper "Approximating the Graph Edit Distance with Compact Neighborhood Representations"

def build_nt(graph, root, height, k):
    """
    * builds a neighborhood tree of a graph with a given root node, height and k

    * param graph: a networkx graph
    * param root: a node in the graph to be the root of the neighborhood tree
    * param height: the height of the neighborhood tree
    * param k: the maximum height difference for redundancy elimination

    * return: a networkx DiGraph object representing the neighborhood tree

    * runtime: O(|E(graph)| * (k+1)) 
    
    * description:
    * The function builds a neighborhood tree of a graph with a given root node, height and k
    * The neighborhood tree is a directed graph with the root node as the root and the children of the nodes as the neighbors
    * The function uses a breadth-first search to build the neighborhood tree
    * The function also checks if the graph is connected and sets the height of the tree accordingly
    * The function uses the canonical encoding of the graph to encode the tree
    """
    if nx.is_connected(graph):
        height = min(height, nx.diameter(graph)+k) 
    tree = nx.DiGraph()
    tree.add_node(root, label=graph.nodes[root]["label"], height=0)
    D = {} # D[v] is the depth of node v
    D[root] = 0
    Phi = {} # Phi[v] is the original node of node v
    Phi[root] = root
    for i in range(1, height):
        F = {} # F[v] is the node in the tree that corresponds to node v
        for v in sorted(tree.nodes):
            if tree.out_degree(v) == 0: # if v is a leaf
                for u in graph.neighbors(Phi[v]):
                    if u not in D:
                        D[u] = i
                    if D[u] + k >= i: # check if the node already apeared within the last k levels
                        # add the node to the tree if it is not already in the tree
                        if u not in F:
                            c = u
                            tree.add_node(c, label=graph.nodes[c]["label"], height=i)
                            Phi[c] = u
                            F[u] = c
                        # add the edge to the tree 
                        tree.add_edge(v, F[u], label=graph.edges[v, F[u]]["label"])
    # add the encoding of the tree to the graph
    tree = tree.to_undirected() # convert the tree to an undirected graph
    tree.graph["encoding"] = encode_graph(tree)
    tree.graph["id"] = graph.graph["id"]

    return tree

def sdted(treee1, treee2, subgraph_dict1, subgraph_dict2, cache):
    """
    * calculates the structure and depth preserving tree edit distance (SDTED) between two trees

    * param treee1: a networkx DiGraph object representing the first tree, in this case a neighborhood tree
    * param treee2: a networkx DiGraph object representing the second tree, in this case a neighborhood tree
    * param subgraph_dict1: a dictionary containing the subgraphs of the nodes of the first tree
    * param subgraph_dict2: a dictionary containing the subgraphs of the nodes of the second tree

    * return: the SDTED between the two trees

    * description:
    * The function calculates the structure and depth preserving tree edit distance (SDTED) between two trees in a recursive manner using a cache to store the results of the calculations
    """
    def pad(tree, number):
        """
        * pads a tree with undefined nodes to a given number of children of the root

        * param tree: a networkx DiGraph object representing the tree to be padded
        * param number: the number of children the root should have

        * return: the padded tree

        * runtime: O(number)

        * description:
        * The function pads a tree with undefined nodes to a given number of children of the root
        * The padding is done by adding undefined nodes to the tree with the label "pad" until the root has the given number of children
        """
        root = next(iter(tree.nodes)) # get the root of the tree 
        # check if the root already has the desired number of children
        current_degree = tree.degree[root]
        if current_degree >= number:
            return tree

        missing_children = number - current_degree # calculate the number of missing children
        new_node = max(tree.nodes) + 1 # get the next available node identifier
        # create the new nodes and edges
        new_nodes = [(new_node + i, {"label": "pad", "height": tree.nodes[root]["height"] + 1}) for i in range(missing_children)]
        new_edges = [(root, new_node + i) for i in range(missing_children)]
        # add the new nodes and edges to the tree
        tree.add_nodes_from(new_nodes)
        tree.add_edges_from(new_edges)

        return tree

    def recursive_sdted(tree1, tree2, depth):
        """
        * calculates the SDTED between two trees recursively

        * param tree1: a networkx DiGraph object representing the first tree, in this case a neighborhood tree or a subgraph of a neighborhood tree
        * param tree2: a networkx DiGraph object representing the second tree, in this case a neighborhood tree or a subgraph of a neighborhood tree
        * param depth: the depth of the recursion

        * return: the SDTED between the two trees

        * description:
        * The function calculates the SDTED between two trees recursively in a bottom-up manner
        """
        w = 1 # weight for the depth of the tree
        # check if the calculation is already in the cache, if so return the result
        key = (tree1.graph["encoding"], tree2.graph["encoding"])
        if key in cache:
            return cache[key] #* pow(base=w, exp=depth)

        # n is the maximum number of children of the roots of the two trees
        children1 = sorted(tree1.neighbors(list(tree1.nodes)[0]))
        children2 = sorted(tree2.neighbors(list(tree2.nodes)[0]))
        n = max(len(children1), len(children2))

        # add undefined nodes to the trees, if the roots have a different number of children

        if len(children1) != len(children2):
            tree1_padded = pad(tree1.copy(), n)
            tree2_padded = pad(tree2.copy(), n)
        else:
            tree1_padded = tree1
            tree2_padded = tree2

        root1 = list(tree1_padded.nodes())[0] # get the root of the first tree
        root2 = list(tree2_padded.nodes())[0] # get the root of the second tree

        # create the cost matrix as n x n matrix
        cost_matrix = np.zeros((n, n))

        # get the children of the roots (ensure that we use a list, which does not change the order of the children)
        children1 = sorted(tree1_padded.neighbors(root1))  
        children2 = sorted(tree2_padded.neighbors(root2))  

        # calculate the cost of the children of the roots 
        for i in range(n):
            for j in range(n):
                
                child1_ident = children1[i] # get the identifier of the child of the first tree
                child2_ident = children2[j] # get the identifier of the child of the second tree

                child1 = tree1_padded.nodes[child1_ident] # get the node of the child of the first tree
                child2 = tree2_padded.nodes[child2_ident] # get the node of the child of the second tree
                
                child1_label = child1["label"] # get the label of the child of the first tree
                child2_label = child2["label"] # get the label of the child of the second tree

                if child1_label != "pad" or child2_label != "pad": # check if at least one of the children is not a dummy node
                    if child1_label != "pad" and child2_label == "pad": # child1 exists, child2 is a dummy node -> insert child1
                        cost_matrix[i][j] = (tree1_padded.nodes[child1_ident]["cost"]) * pow(base=w, exp=depth)
                        #// cost_matrix[i][j] = (tree1_padded.nodes[child1_ident]["cost"] + cost_insert_edge(tree1_padded.edges[root1, child1_ident]["label"])) * (1/(1+depth+1))
                    elif child2_label != "pad" and child1_label == "pad": # child2 exists, child1 is a dummy node -> insert child2
                        cost_matrix[i][j] = (tree2_padded.nodes[child2_ident]["cost"]) * pow(base=w, exp=depth)
                        #// cost_matrix[i][j] = (tree2_padded.nodes[child2_ident]["cost"] + cost_insert_edge(tree2_padded.edges[root2, child2_ident]["label"])) * (1/(1+depth+1))
                    else: # both children are not dummy nodes -> recursive call
                        # check if the edge labels are different and add the cost of relabeling the edge
                        temp_cost = 0
                        if tree1_padded.edges[root1, child1_ident]["label"] != tree2_padded.edges[root2, child2_ident]["label"]:
                            temp_cost = 1 * pow(base=w, exp=depth)
                        #// temp_cost = cost_relabel_edge(tree1_padded.edges[root1, child1_ident]["label"], tree2_padded.edges[root2, child2_ident]["label"]) if tree1_padded.edges[root1, child1_ident]["label"] != tree2_padded.edges[root2, child2_ident]["label"] else 0
                        # check cache for recursive call
                        if (subgraph_dict1[child1_ident].graph["encoding"], subgraph_dict2[child2_ident].graph["encoding"]) in cache:
                            recursive_cost = cache[(subgraph_dict1[child1_ident].graph["encoding"], subgraph_dict2[child2_ident].graph["encoding"])] 
                        else:
                            # recursive call on the subgraphs induced by the children of the roots
                            recursive_cost = recursive_sdted(subgraph_dict1[child1_ident], subgraph_dict2[child2_ident], depth + 1)
                        # calculate the cost of the recursive call and add the cost of the edge relabeling 
                        cost_matrix[i][j] = (recursive_cost + temp_cost) 

        # calculate the cost of the roots
        cost_root = 0 
        if tree1_padded.nodes[root1]["label"] != tree2_padded.nodes[root2]["label"]: # check if the root labels are different
            cost_root = 1 * pow(base=w, exp=depth)
            #// cost_root = cost_relabel_node(tree1_padded.nodes[root1]["label"], tree2_padded.nodes[root2]["label"])
        
        # use the Hungarian Algorithm to find the optimal matching of the children
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # calculate the cost of the matching
        cost = 0
        for i in range(n): 
            cost += cost_matrix[row_ind[i]][col_ind[i]]
        
        # calculate the final result of the SDTED
        result = (cost + cost_root) 
        # add the result to the cache
        cache[key] = result

        return result

    # start the recursive calculation of the SDTED between the two trees 
    result = recursive_sdted(treee1, treee2, 0)
    # add the result to the cache
    cache[(treee1.graph["encoding"], treee2.graph["encoding"])] = result

    return result

# ? The following functions are used to :
# ? 1. frontload the costs of insertion and deletion of nodes in the neighborhood trees
# ? 2. create a subgraph of the neighborhood tree, starting from a specific node
# ? 3. create a dictionary of subgraphs for each node in the neighborhood tree
# ? 4. create a dictionary of neighborhood trees for each node in the graph
# ? while this takes a lot of time upfront, it makes the calculations of the SDTED much faster

def calculate_costs(tree):
    """
    * calculates the costs of inserting and deleting nodes in a neighborhood tree and stores them in the nodes

    * param tree: a networkx DiGraph object representing the neighborhood tree

    * return: the neighborhood tree with the costs of inserting and deleting nodes stored in the nodes

    * runtime: O(|V(tree)| + |E(tree)|)

    * description:
    * The function calculates the costs of inserting and deleting nodes in a neighborhood tree based on their subtree via a breadth-first search
    """
    for node in sorted(tree.nodes):
        cost = 1
        queue = deque([node])
        visited = set([node])

        # use a breadth-first search to calculate the cost of inserting and deleting nodes
        while queue:
            current_node = queue.popleft()
            for neighbor in tree.neighbors(current_node):
                if neighbor not in visited and tree.nodes[neighbor]["height"] > tree.nodes[current_node]["height"]:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    cost += 1
                    #// cost += cost_insert_edge(tree.edges[current_node, neighbor]["label"]) + cost_insert_node(tree.nodes[neighbor]["label"])
        # add the cost to the node
        tree.nodes[node]["cost"] = cost

    return tree


def create_subgraph(graph, node):
    """
    * creates a subgraph of a graph starting from a specific node

    * param graph: a networkx DiGraph object representing the graph
    * param node: the node in the graph to start the subgraph from

    * return: a networkx DiGraph object representing the subgraph

    * runtime: O(|V(graph)| + |E(graph)|)

    * description:
    * The function creates a subgraph of a graph starting from a specific node by adding all nodes and edges that are reachable from the node
    * The function preserves the structure of the graph and all node and edge attributes
    """
    subgraph = nx.DiGraph()
    subgraph.add_node(node, label=graph.nodes[node]["label"], height=graph.nodes[node]["height"], cost=graph.nodes[node]["cost"])

    current_level = [node]
    next_level = []

    # use a breadth-first search to create the subgraph 
    while current_level:
        for current_node in current_level:
            for neighbor in graph.neighbors(current_node):
                if graph.nodes[neighbor]["height"] > graph.nodes[current_node]["height"]: # check if the neighbor is a child of the current node
                    if neighbor not in subgraph.nodes:
                        subgraph.add_node(neighbor, label=graph.nodes[neighbor]["label"], height=graph.nodes[neighbor]["height"], cost=graph.nodes[neighbor]["cost"])
                        subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])
                        next_level.append(neighbor)
                    else:
                        subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])
        current_level = next_level
        next_level = []

    # add the encoding of the subgraph to the graph
    subgraph = subgraph.to_undirected() # convert the subgraph to an undirected graph
    subgraph.graph["encoding"] = encode_graph(subgraph)
    subgraph.graph["id"] = graph.graph["id"]

    return subgraph


def create_subgraph_dict(graph):
    """
    * creates a dictionary of subgraphs for each node in a graph

    * param graph: a networkx DiGraph object representing the graph

    * return: a dictionary containing the subgraphs of the nodes of the graph

    * runtime: O(|V(graph)| * (|V(graph)| + |E(graph)|))

    * description:
    * The function creates a dictionary of subgraphs for each node in a graph by calling the create_subgraph function for each node
    """
    subgraph_dict = {}
    for node in sorted(graph.nodes):
        subgraph_dict[node] = create_subgraph(graph, node)
    return subgraph_dict


def create_nt_dict(graphs, height, k):
    """
    * creates a dictionary of neighborhood trees with their subgraph dictionary for each node in a graph

    * param graphs: a dictionary containing networkx Graph objects representing the graphs
    * param height: the height of the neighborhood trees
    * param k: the maximum height difference for redundancy elimination

    * return: a dictionary containing the neighborhood trees with their subgraph dictionary for each node in a graph

    * description:
    * The function creates a dictionary of neighborhood trees with their subgraph dictionary for each node in a graph by calling the build_nt and create_subgraph_dict functions for each node
    """
    nt_dict = {}
    for graph_id in graphs:
        for node in sorted(graphs[graph_id].nodes):
            nt = build_nt(graphs[graph_id], node, height, k)
            nt_dict[(graph_id, node)] = calculate_costs(nt), create_subgraph_dict(nt) # use graph_id and node as key for the dictionary
    return nt_dict

# ? The following functions are used to calculate the cost matrix, edit paths and matchings between the graphs

def derive_edit_path(graph1, graph2, row_ind, col_ind):
    """
    * derives the edit path between two graphs based on the Hungarian Algorithm matching

    * param graph1: a networkx DiGraph object representing the first graph
    * param graph2: a networkx DiGraph object representing the second graph
    * param row_ind: the row indices of the Hungarian Algorithm matching
    * param col_ind: the column indices of the Hungarian Algorithm matching

    * return: the edit path between the two graphs

    * runtime: O(|V(graph1)| + |E(graph1)| + |V(graph2)| + |E(graph2)|)

    * description:
    * The function derives the edit path between two graphs based on the Hungarian Algorithm matching
    * A node is relabeled if its in the matching and the labels are different
    * A node is deleted if it is not in the matching and in the first graph
    * A node is inserted if it is not in the matching and in the second graph
    * An edge is relabeled if its in the matching and the labels are different
    * An edge is deleted if it is not in the matching and in the first graph
    * An edge is inserted if it is not in the matching and in the second graph
    """
    def normalize_edge(edge):
        return tuple(sorted(edge))

    edit_path = []
    #// edit_cost = 0
    nodes1 = list(graph1.nodes)
    nodes2 = list(graph2.nodes)

    visited_edges = set()
    
    # create a mapping of the matched nodes
    mapping = {}
    mapping_inv = {}
    matched_nodes1 = set()
    matched_nodes2 = set()

    for i, j in zip(row_ind, col_ind):
        if i < len(nodes1) and j < len(nodes2):  # Nur echte Knotenzuordnungen berücksichtigen
            node1 = nodes1[i]
            node2 = nodes2[j]
            mapping[node1] = node2
            mapping_inv[node2] = node1
            matched_nodes1.add(node1)
            matched_nodes2.add(node2)
            
            if graph1.nodes[node1]["label"] != graph2.nodes[node2]["label"]:
                edit_path.append(f"relabel node: {node1} -> {node2}")
                #// edit_cost += cost_relabel_node(graph1.nodes[node1]["label"], graph2.nodes[node2]["label"])
    
    # delete unmatched nodes in graph1
    unmatched_nodes1 = set(graph1.nodes) - matched_nodes1
    for node in unmatched_nodes1:
        for neighbor in graph1.neighbors(node):
            if normalize_edge((node, neighbor)) not in visited_edges:
                edit_path.append(f"delete edge: {node}-{neighbor}")
                #// edit_cost += cost_delete_edge(graph1.edges[node, neighbor]["label"])
                visited_edges.add(normalize_edge((node, neighbor)))
        edit_path.append(f"delete node: {node}")
        #// edit_cost += cost_delete_node(graph1.nodes[node]["label"])
    
    # insert unmatched nodes in graph2
    unmatched_nodes2 = set(graph2.nodes) - matched_nodes2
    for node in unmatched_nodes2:
        edit_path.append(f"insert node: {node}")
        #// edit_cost += cost_insert_node(graph2.nodes[node]["label"])

    for node in unmatched_nodes2:
        for neighbor in graph2.neighbors(node):
                if neighbor not in mapping_inv: # check if the neighbor is not in the mapping
                    if normalize_edge((node, neighbor)) not in visited_edges:
                        edit_path.append(f"insert n_edge: {node}-{neighbor}")
                        #// edit_cost += cost_insert_edge(graph2.edges[node, neighbor]["label"])
                        visited_edges.add(normalize_edge((node, neighbor)))
                elif neighbor in mapping_inv: # check if the neighbor is in the mapping
                    if normalize_edge((node, mapping_inv[neighbor])) not in visited_edges:
                        edit_path.append(f"insert h_edge: {node}-{mapping_inv[neighbor]}")
                        #// edit_cost += cost_insert_edge(graph2.edges[node, mapping_inv[neighbor]]["label"])
                        visited_edges.add(normalize_edge((node, mapping_inv[neighbor])))

    # delete edges in graph1 which are not in the matching and relabel edges which are in the matching but have different labels
    for (u,v) in graph1.edges:
        if u not in mapping or v not in mapping:
            continue
        if normalize_edge((u,v)) in visited_edges or normalize_edge((mapping[u], mapping[v])) in visited_edges:
            continue
        if (mapping[u], mapping[v]) not in graph2.edges:
            edit_path.append(f"delete edge: {u}-{v}")
            #// edit_cost += cost_delete_edge(graph1.edges[u,v]["label"])
            visited_edges.add(normalize_edge((u,v)))
            visited_edges.add(normalize_edge((mapping[u], mapping[v])))
        elif graph1.edges[u,v]["label"] != graph2.edges[mapping[u], mapping[v]]["label"]:
            edit_path.append(f"relabel edge: {u}-{v} -> {mapping[u]}-{mapping[v]}")
            #// edit_cost += cost_relabel_edge(graph1.edges[u,v]["label"], graph2.edges[mapping[u], mapping[v]]["label"])
            visited_edges.add(normalize_edge((u,v)))
            visited_edges.add(normalize_edge((mapping[u], mapping[v])))

    # insert edges in graph2 which are not in the matching and relabel edges which are in the matching but have different labels
    for (u,v) in graph2.edges:
        if u not in mapping_inv or v not in mapping_inv:
            continue
        if normalize_edge((u,v)) in visited_edges or normalize_edge((mapping_inv[u], mapping_inv[v])) in visited_edges:
            continue
        if (mapping_inv[u], mapping_inv[v]) not in graph1.edges:
            edit_path.append(f"insert edge: {mapping_inv[u]}-{mapping_inv[v]}")
            #// edit_cost += cost_insert_edge(graph2.edges[u,v]["label"])
            visited_edges.add(normalize_edge((u,v)))
            visited_edges.add(normalize_edge((mapping_inv[u], mapping_inv[v])))
    
    return edit_path #// , edit_cost


def calculate_GED_cnt(graph1, graph2, nt_dict = None, cache = {}, height=8, k=0):
    """
    * calculates the Graph Edit Distance (GED) between two graphs using the Hungarian Algorithm based on the SDTED as the cost function

    * param graph1: a networkx Graph object representing the first graph
    * param graph2: a networkx Graph object representing the second graph
    * param nt_dict: a dictionary containing the neighborhood trees with their subgraph dictionary for each node in a graph
    * param cache: a dictionary containing the results of the SDTED calculations

    * return: the row indices, column indices, the minimum GED, the edit path and the matching between the two graphs

    * description:
    * The function calculates the Graph Edit Distance (GED) between two graphs using the Hungarian Algorithm based on the SDTED as the cost function
    * The function creates a cost matrix based on the SDTED between the neighborhood trees of the nodes of the two graphs
    * The SDTED is used as the cost function for the Hungarian Algorithm
    * The function uses the Hungarian Algorithm to find the optimal matching of the nodes
    * The function derives the edit path between the two graphs based on the Hungarian Algorithm matching and calculates the minimum GED
    """
    if nt_dict is None:
        nt_dict = create_nt_dict({graph1.graph["id"]: graph1, graph2.graph["id"]: graph2}, height, k)
    
    n1, n2 = len(graph1.nodes), len(graph2.nodes)
    cost_matrix = np.full((n1, n2), np.inf)  # initialize the cost matrix with infinity

    nodes1 = list(graph1.nodes)
    nodes2 = list(graph2.nodes)

    for i, node1 in enumerate(nodes1):
        for j, node2 in enumerate(nodes2):
            nt1 = nt_dict[(graph1.graph["id"], node1)][0]
            nt2 = nt_dict[(graph2.graph["id"], node2)][0]
            nt1_subgraph = nt_dict[(graph1.graph["id"], node1)][1]
            nt2_subgraph = nt_dict[(graph2.graph["id"], node2)][1]

            # calculate the SDTED between the neighborhood trees of the nodes
            current_result = sdted(nt1, nt2, nt1_subgraph, nt2_subgraph, cache)
            cost_matrix[i, j] = current_result

    # calculate the Hungarian Algorithm matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # get a list of matched nodes
    matching = [(nodes1[i], nodes2[j]) for i, j in zip(row_ind, col_ind)]
    # calculate the edit path between the two graphs and the minimum GED
    edit_path = derive_edit_path(graph1, graph2, row_ind, col_ind)
    min_GED = len(edit_path)

    return row_ind, col_ind, min_GED, edit_path, matching


def calculate_cost_matrix(graphs, height=8, k=0):
    """
    * calculates the GED cost matrix between a set of graphs using the Hungarian Algorithm based on the SDTED as the cost function

    * param graphs: a dictionary containing networkx Graph objects representing the graphs
    * param height: the height of the neighborhood trees
    * param k: the maximum height difference for redundancy elimination

    * return: the GED cost matrix between the graphs, the edit paths and the matchings between the graphs

    * description:
    * The function calculates the GED cost matrix between a set of graphs using the Hungarian Algorithm based on the SDTED as the cost function
    * The function uses concurrent.futures to parallelize the calculation of the GED cost matrix
    """
    basetime = t.time()
    cache = {}
    # create the NT dictionary
    nt_dict = create_nt_dict(graphs, height, k)
    print(f"Create the NT dictionary: {t.time() - basetime}s")
    basetime = t.time()

    # create the cost matrix, edit paths and matchings
    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0)
    edit_paths = {}
    matchings = {}

    for i, j in combinations(range(len(graph_ids)), 2):
        row_ind, col_ind, min_GED, edit_path, matching = calculate_GED_cnt(graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict, cache)
        cost_matrix[i, j] = cost_matrix[j, i] = min_GED
        edit_paths[(i, j)] = edit_paths[(j, i)] = edit_path
        matchings[(i, j)] = matchings[(j, i)] = matching

    print(f"Calculating the cost matrix: {t.time() - basetime}s")
    print("\n")
    return cost_matrix, edit_paths, matchings

def nx_cost_matrix(graphs, n_iter=2):

    basetime = t.time()

    # create the cost matrix, edit paths and matchings
    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0)

    # use concurrent.futures to parallelize the calculation of the GED cost matrix
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        # calculate the GED between all pairs of graphs
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(nx.optimize_graph_edit_distance, graphs[graph_ids[i]], graphs[graph_ids[j]], node_match=node_match, edge_match=edge_match)

        # get the results of the futures and store them in the cost matrix, edit paths and matchings
        for (i, j), future in futures.items():
            ged_iter = future.result()
            approx_ged = 999
            for idx, ged in enumerate(ged_iter):
                if idx == n_iter:  
                    approx_ged = ged
                    break
            cost_matrix[i, j] = cost_matrix[j, i] = approx_ged


    print(f"Calculating the cost matrix: {t.time() - basetime}s")
    print("\n")
    return cost_matrix

def cnt_bound_matrix(graphs,cnt_matrix = None, height=5, k=0):
    if cnt_matrix is None:
        cnt_matrix = calculate_cost_matrix(graphs, height, k)[0]

    basetime = t.time()
    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0)

    # use concurrent.futures to parallelize the calculation of the GED cost matrix
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        # calculate the GED between all pairs of graphs
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(nx.optimize_graph_edit_distance, graphs[graph_ids[i]], graphs[graph_ids[j]], node_match=node_match, edge_match=edge_match, upper_bound=cnt_matrix[i,j])

        # get the results of the futures and store them in the cost matrix, edit paths and matchings
        for (i, j), future in futures.items():
            ged_iter = future.result()
            for idx, ged in enumerate(ged_iter): 
                approx_ged = ged
                break
            cost_matrix[i, j] = cost_matrix[j, i] = approx_ged

    print(f"Calculating the cost matrix: {t.time() - basetime}s")
    print("\n")
    return cost_matrix


# ? The following functions are used to calculate the GED based on a standard BGM approach

def standard_bgm(graph1, graph2):
    """
    * calculates the Graph Edit Distance (GED) between two graphs using the Hungarian Algorithm based on the standard cost function

    * param graph1: a networkx Graph object representing the first graph
    * param graph2: a networkx Graph object representing the second graph

    * return: the row indices, column indices, the minimum GED, the edit path and the matching between the two graphs

    * description:
    * The function calculates the Graph Edit Distance (GED) between two graphs using the Hungarian Algorithm based on the standard cost function
    * The usual heuristic of BGM is extended by the number of neighbors of the nodes
    """
    nodes1 = list(graph1.nodes)
    nodes2 = list(graph2.nodes)

    n1, n2 = len(nodes1), len(nodes2)
    cost_matrix = np.full((n1, n2), np.inf)  # Initialisiere hohe Kosten
    
    for i, u in enumerate(nodes1):
        for j, v in enumerate(nodes2):
            label_cost = 1 if graph1.nodes[u]["label"] != graph2.nodes[v]["label"] else 0
            neighbor_cost = abs(len(list(graph1.neighbors(u))) - len(list(graph2.neighbors(v))))
            cost_matrix[i, j] = label_cost + neighbor_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching = [(nodes1[i], nodes2[j]) for i, j in zip(row_ind, col_ind)]
    edit_path = derive_edit_path(graph1, graph2, row_ind, col_ind)

    min_ged = len(edit_path)

    return row_ind, col_ind, min_ged, edit_path, matching

def standard_bgm_matrix(graphs):
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
    edit_paths = {}
    matchings = {}

    # use concurrent.futures to parallelize the calculation of the GED cost matrix
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        # calculate the GED between all pairs of graphs
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(standard_bgm, graphs[graph_ids[i]], graphs[graph_ids[j]])

        # get the results of the futures and store them in the cost matrix, edit paths and matchings
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
    """
    * applies the edit path to the first graph based on the matching

    * param graph1: a networkx Graph object representing the first graph
    * param graph2: a networkx Graph object representing the second graph
    * param edit_path: the edit path between the two graphs as a list of actions
    * param matching: the matching between the nodes of the two graphs

    * return: the first graph after applying the edit path

    * runtime: O(edit_path * O(action))

    * description:
    * The function applies the edit path to the first graph based on the matching
    * The function relabels nodes and edges, deletes nodes and edges, and inserts nodes and edges based on the edit
    """
    # create a copy of the first graph
    graph1 = graph1.copy()

    # iterate over the actions in the edit path
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
    """
    * checks if two graphs are isomorphic based on the node and edge labels

    * param graph1: a networkx Graph object representing the first graph
    * param graph2: a networkx Graph object representing the second graph

    * return: True if the graphs are isomorphic, False otherwise

    * description:
    * The function checks if two graphs are isomorphic based on the node and edge labels
    * The function uses the is_isomorphic function from NetworkX with custom node and edge matching functions
    """
    # check if the number of nodes and edges is the same, if not return False
    if len(graph1.nodes) != len(graph2.nodes) or len(graph1.edges) != len(graph2.edges):
        return False

    # check if the graphs are isomorphic based on the node and edge labels using the is_isomorphic function
    return nx.is_isomorphic(graph1, graph2, node_match=node_match, edge_match=edge_match)

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


def print_two_graphs(graph1, graph2, layout='spring'):
    """
    * prints two graphs side by side with their node and edge labels

    * param graph1: a networkx Graph object representing the first graph
    * param graph2: a networkx Graph object representing the second graph
    * param layout: the layout type for the graphs

    * return: None
    """

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

    # Create a dictionary that combines node IDs and labels
    node_labels1 = {node: f"{node}: {label}" for node, label in nx.get_node_attributes(graph1, 'label').items()}
    node_labels2 = {node: f"{node}: {label}" for node, label in nx.get_node_attributes(graph2, 'label').items()}
    
    # Plot graph1
    edge_labels1 = nx.get_edge_attributes(graph1, 'label')
    nx.draw(graph1, pos1, with_labels=True, labels=node_labels1, ax=axes[0])
    nx.draw_networkx_edge_labels(graph1, pos1, edge_labels=edge_labels1, ax=axes[0])
    axes[0].set_title("Graph " + str(graph1_label))
    
    # Plot graph2
    edge_labels2 = nx.get_edge_attributes(graph2, 'label')
    nx.draw(graph2, pos2, with_labels=True, labels=node_labels2, ax=axes[1])
    nx.draw_networkx_edge_labels(graph2, pos2, edge_labels=edge_labels2, ax=axes[1])
    axes[1].set_title("Graph " + str(graph2_label))
    
    plt.show()
    
    return None


def load_graphs(dataset, n = None):
    """
    * loads the graphs from a dataset

    * param dataset: the name of the dataset
    * param n: the number of graphs to load from the dataset

    * return: a dictionary containing the graphs
    """
    # get the path to the dataset
    current_dir = os.path.dirname(__file__)
    dataset_name = str(dataset)
    path = os.path.join(current_dir, "data", dataset_name)

    # get the edges of the graph from the adjacency matrix
    edges = pd.read_csv(f"{path}\{dataset_name}_A.txt", header=None, sep=",")
    edges.columns = ["source", "target"]


    # get the node to graph mapping
    node_to_graph = pd.read_csv(f"{path}\{dataset_name}_graph_indicator.txt", header=None)
    node_to_graph.columns = ["graph_id"]


    # get the graph labels
    graph_labels = pd.read_csv(f"{path}\{dataset_name}_graph_labels.txt", header=None)
    graph_labels.columns = ["label"]


    # get the node labels
    if os.path.exists(f"{path}\{dataset_name}_node_labels.txt"):
        node_labels = pd.read_csv(f"{path}\{dataset_name}_node_labels.txt", header=None)
        node_labels.columns = ["label"]

    # get the edge labels
    if os.path.exists(f"{path}\{dataset_name}_edge_labels.txt"):
        edge_labels = pd.read_csv(f"{path}\{dataset_name}_edge_labels.txt", header=None)
        edge_labels.columns = ["label"]

    # create a dictionary of graphs
    graphs = {}
    if n == None:
        i = node_to_graph["graph_id"].nunique()
    else:
        i = 0
    for graph_id in node_to_graph["graph_id"].unique():
        if i == n:
            break
        graph_nodes = node_to_graph[node_to_graph["graph_id"] == graph_id].index + 1
        subgraph_edges = edges[edges["source"].isin(graph_nodes) & edges["target"].isin(graph_nodes)]
        graphs[graph_id] = nx.from_pandas_edgelist(subgraph_edges, source="source", target="target")
        graphs[graph_id].add_nodes_from(graph_nodes)
        graphs[graph_id].graph["label"] = graph_labels.loc[graph_id - 1, "label"]
        graphs[graph_id].graph["id"] = graph_id

        # add node labels to the graph if they exist
        if "node_labels" in locals():
            for node in graph_nodes:
                graphs[graph_id].nodes[node]["label"] = node_labels.loc[node - 1, "label"]
        else: 
            for node in graph_nodes:
                graphs[graph_id].nodes[node]["label"] = "dummy"

        # add edge labels to the graph if they exist
        if "edge_labels" in locals():
            for _, row in subgraph_edges.iterrows():
                source, target = row["source"], row["target"]
                edge_label = edge_labels.loc[edges[(edges["source"] == source) & (edges["target"] == target)].index[0], "label"]
                graphs[graph_id].edges[source, target]["label"] = edge_label
                graphs[graph_id].edges[target, source]["label"] = edge_label  # Ungerichtete Kante (symmetrisch)
        else:
            for _, row in subgraph_edges.iterrows():
                source, target = row["source"], row["target"]
                graphs[graph_id].edges[source, target]["label"] = "dummy"
                graphs[graph_id].edges[target, source]["label"] = "dummy"
        i += 1
    return graphs

import seaborn as sns

def plot_cost_matrix(cost_matrix, title="Cost Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cost_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
    plt.xlabel('Graph 1 Nodes')
    plt.ylabel('Graph 2 Nodes')
    plt.title(title)
    plt.show()
