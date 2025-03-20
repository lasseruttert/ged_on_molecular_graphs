import matplotlib.pyplot as plt
import numpy as np
import main as main
import seaborn as sns
import graph_classification as gc
from bgm import graph_edit_distance_bipartite

# Assuming cost_matrix is your matrix
def plot_cost_matrix(cost_matrix, title="Cost Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cost_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
    plt.xlabel('Graph 1 Nodes')
    plt.ylabel('Graph 2 Nodes')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    n = 40

    graphs = main.load_graphs("MUTAG", n)

    graph1 = graphs[1]
    graph1_nt = main.build_nt(graph1,1, 8, 3)
    main.print_two_graphs(graph1, graph1_nt)

    cost_matrix, edit_matrix, matchings = main.calculate_cost_matrix(graphs, 1, 0)

    # plot_cost_matrix(cost_matrix, title="Cost Matrix")

    # for i in range(n): 
    #     for j in range(n):
    #         if i == j:
    #             continue
    #         if i > j:
    #             continue
    #         new_graph = main.graph_matcher(graphs[i+1], graphs[j+1], edit_matrix[i,j], matchings[i,j])
    #         current_bool = main.isomorph_check(graphs[j+1], new_graph)
    #         if not current_bool:
    #             print(edit_matrix[i,j])
    #             print(matchings[i,j])
    #             main.print_two_graphs(graphs[j+1], new_graph)
    #             print(i+1, j+1)
    #             print("\n")
    #             print("\n")

    bgm_cost_matrix, bgm_edit_matrix, bgm_matchings = main.standard_bgm_matrix(graphs)

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

    print(cost_matrix - bgm_cost_matrix)
    print(np.average(cost_matrix - bgm_cost_matrix))



    print("Done")
