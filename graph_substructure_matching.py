import networkx as nx
import main as main
from itertools import combinations

# ! GED cannot be used as a similarity measure for subgraph matching

def create_subgraph(graph, nodes):
    subgraph = nx.Graph()
    subgraph.graph["id"] = graph.graph["id"]
    for node in nodes:
        subgraph.add_node(node, label=graph.nodes[node]["label"])
    for source, target in graph.edges:
        if source in nodes and target in nodes:
            subgraph.add_edge(source, target, label=graph.edges[source, target]["label"])
    return subgraph

def create_all_subgraphs(graph):
    all_subgraphs = []
    for i in range(1, len(graph.nodes) + 1):
        for subgraph_nodes in combinations(graph.nodes, i):
            all_subgraphs.append(create_subgraph(graph, subgraph_nodes))
    return all_subgraphs

def create_benzol_ring():
    ring = nx.Graph()
    ring.graph["id"] = "benzol"
    ring.add_node(1, label="0")
    ring.add_node(2, label="0")
    ring.add_node(3, label="0")
    ring.add_node(4, label="0")
    ring.add_node(5, label="0")
    ring.add_node(6, label="0")

    ring.add_edge(1, 2, label="0")
    ring.add_edge(2, 3, label="0")
    ring.add_edge(3, 4, label="0")
    ring.add_edge(4, 5, label="0")
    ring.add_edge(5, 6, label="0")
    ring.add_edge(6, 1, label="0")
    return ring

if __name__ == "__main__":
    # matched_substructures = {}
    # subgraphs = create_all_subgraphs(graphs[1])

    ring = create_benzol_ring()

    # i = 0

    # for subgraph in subgraphs:
    #     # check if subgraph is connected
    #     if nx.is_connected(subgraph):
    #         _,_,GED,_,_ = main.calculate_GED_bgm(subgraph, ring)
    #         matched_substructures[i] = (GED,subgraph)
    #         i += 1

    # sorted_matches = dict(sorted(matched_substructures.items(), key=lambda item: item[1][0]))
    # min_key, (min_ged, min_subgraph) = min(matched_substructures.items(), key=lambda item: item[1][0])

    # main.print_two_graphs(min_subgraph, ring)

    graphs = main.load_graphs("MUTAG",1)
    nodes1 = [1,2,3,4,5,6]
    nodes2 = [4,5,7,8,9,10]
    nodes3 = [9,10,11,12,13,14]

    # main.print_two_graphs(graphs[1],ring)

    subgraph1 = create_subgraph(graphs[1],nodes1)
    subgraph2 = create_subgraph(graphs[1],nodes2)
    subgraph3 = create_subgraph(graphs[1],nodes3)

    main.print_two_graphs(subgraph1,ring)
    main.print_two_graphs(subgraph2,ring)
    main.print_two_graphs(subgraph3,ring)
    
    _,_,ged1,e1,m1 = main.calculate_GED_bgm(subgraph1, ring)
    _,_,ged2,e2,m2 = main.calculate_GED_bgm(subgraph2, ring)
    _,_,ged3,e3,m3 = main.calculate_GED_bgm(subgraph3, ring)

    g1 = main.graph_matcher(subgraph1,ring,e1,m1)
    g2 = main.graph_matcher(subgraph2,ring,e2,m2)
    g3 = main.graph_matcher(subgraph3,ring,e3,m3)

    print(main.isomorph_check(g1,ring))
    print(main.isomorph_check(g2,ring))
    print(main.isomorph_check(g3,ring))

    print(ged1)
    print(ged2)
    print(ged3)

    # _,_,ged_bgm1,_,_ = main.standard_bgm(subgraph1, ring)
    # _,_,ged_bgm2,_,_ = main.standard_bgm(subgraph2, ring)
    # _,_,ged_bgm3,_,_ = main.standard_bgm(subgraph3, ring)

    # print(ged_bgm1)
    # print(ged_bgm2)
    # print(ged_bgm3)


    print("Done!")