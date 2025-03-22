import main as main
import time as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from graph_classification import train_kNN, test_knn, knn_matrix
from graph_clustering import cluster_graphs, spectral_clustering, agglomerative_clustering, k_metoid_clustering
import pandas as pd
from bgm import bgm_matrix

if __name__ == "__main__":

    heights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # # ? Costmatrix

    # # * MUTAG - Full - CNT
    # mutag_full = main.load_graphs("MUTAG")
    # mutag_matrix,_,_ = main.calculate_cost_matrix(mutag_full, height=5)
    # np.savetxt(f"MUTAG_Matrix_cnt_5.csv", mutag_matrix, delimiter=",",fmt="%d")
    # plt.figure(figsize=(10, 8))
    # # sns.heatmap(mutag_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
    # # plt.xlabel('Graphs')
    # # plt.ylabel('Graphs')
    # # plt.title("MUTAG - Full - CNT")
    # # plt.savefig("MUTAG_full_cost_matrix.png")
    # plt.clf()

    # mutag_matrix = bgm_matrix(mutag_full)
    # np.savetxt(f"MUTAG_Matrix_bgm_leon.csv", mutag_matrix, delimiter=",",fmt="%d")

    # mutag_matrix = main.nx_cost_matrix(mutag_full, 0)
    # np.savetxt(f"MUTAG_Matrix_nx_0.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix = main.nx_cost_matrix(mutag_full, 1)
    # np.savetxt(f"MUTAG_Matrix_nx_1.csv", mutag_matrix, delimiter=",",fmt="%d")

    # mutag_matrix = main.nx_cost_matrix(mutag_full, 2)
    # np.savetxt(f"MUTAG_Matrix_nx_2.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.standard_bgm_matrix(mutag_full)
    # np.savetxt(f"MUTAG_Matrix_bgm.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=1)
    # np.savetxt(f"MUTAG_Matrix_cnt_1.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=2)
    # np.savetxt(f"MUTAG_Matrix_cnt_2.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=3)
    # np.savetxt(f"MUTAG_Matrix_cnt_3.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=4)
    # np.savetxt(f"MUTAG_Matrix_cnt_4.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=6)
    # np.savetxt(f"MUTAG_Matrix_cnt_6.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=7)
    # np.savetxt(f"MUTAG_Matrix_cnt_7.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=8)
    # np.savetxt(f"MUTAG_Matrix_cnt_8.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=9)
    # np.savetxt(f"MUTAG_Matrix_cnt_9.csv", mutag_matrix, delimiter=",",fmt="%d")
    
    # mutag_matrix, _, _ = main.calculate_cost_matrix(mutag_full, height=10)
    # np.savetxt(f"MUTAG_Matrix_cnt_10.csv", mutag_matrix, delimiter=",",fmt="%d")
    # print("MUTAG - Full - CNT: Done")

    # # * MUTAG - 20 - CNT
    # mutag_20 = main.load_graphs("MUTAG", 20)
    # mutag_20_cnt,_,_ = main.calculate_cost_matrix(mutag_20, height=7)
    # np.savetxt(f"MUTAG_20_cost_matrix.csv", mutag_20_cnt, delimiter=",",fmt="%d")
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(mutag_20_cnt, annot=True, fmt="d", cmap="plasma", cbar=True)
    # plt.xlabel('Graphs')
    # plt.ylabel('Graphs')
    # plt.title("MUTAG - 20 - CNT")
    # plt.savefig("MUTAG_20_cost_matrix.png")
    # plt.clf()
    # print("MUTAG - 20 - CNT: Done")

    # # * MUTAG - 20 - BGM
    # mutag_20 = main.load_graphs("MUTAG", 20)
    # mutag_20_bgm,_,_ = main.standard_bgm_matrix(mutag_20)
    # np.savetxt(f"MUTAG_20_bgm_cost_matrix.csv", mutag_20_bgm, delimiter=",",fmt="%d")
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(mutag_20_bgm, annot=True, fmt="d", cmap="plasma", cbar=True)
    # plt.xlabel('Graphs')
    # plt.ylabel('Graphs')
    # plt.title("MUTAG - 20 - BGM")
    # plt.savefig("MUTAG_20_bgm_cost_matrix.png")
    # plt.clf()
    # print("MUTAG - 20 - BGM: Done")

    # # * MUTAG - 20 - Diff
    # mutag_20_diff = mutag_20_cnt - mutag_20_bgm
    # np.savetxt(f"MUTAG_20_diff_cost_matrix.csv", mutag_20_diff, delimiter=",",fmt="%d")
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(mutag_20_diff, annot=True, fmt="d", cmap="plasma", cbar=True)
    # plt.xlabel('Graphs')
    # plt.ylabel('Graphs')
    # plt.title("MUTAG - 20 - Diff")
    # plt.savefig("MUTAG_20_diff_cost_matrix.png")
    # plt.clf()
    # print("MUTAG - 20 - Diff: Done")


    # # ? Runtime + Precision

    # # * MUTAG - single GED
    # cache = {}
    # mutag_101 = main.load_graphs("MUTAG", 101)
    # runtimes_cnt = []
    # runtimes_bgm = []
    # GEDs_cnt = []
    # GEDs_bgm = []

    # runtime_bgm = 0
    # GED_bgm = 0
    # for i in range(100):
    #     for j in range(100):
    #         basetime = t.time()
    #         GED_bgm += main.standard_bgm(mutag_101[i+1], mutag_101[j+1])[2]
    #         runtime_bgm += t.time() - basetime
    # for height in heights:
    #     runtimes_bgm.append(runtime_bgm / 10000)
    #     GEDs_bgm.append(GED_bgm / 10000)

    # for height in heights:
    #     print(f"Height: {height}")
    #     runtime_cnt = 0
    #     GED_cnt = 0
    #     for i in range(100):
    #         print(f"Run: {i}")
    #         for j in range(100):
    #             basetime = t.time()
    #             GED_cnt += main.calculate_GED_bgm(mutag_101[i+1], mutag_101[j+1], height=height, cache=cache)[2]
    #             runtime_cnt += t.time() - basetime
    #     runtimes_cnt.append(runtime_cnt / 10000)
    #     GEDs_cnt.append(GED_cnt / 10000)

    # plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    # plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title("Runtime of GED calculation: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.savefig("runtime_ged_MUTAG.png")
    # plt.clf()
    # print("MUTAG - single GED: Done")

    # plt.plot(heights, GEDs_cnt, label="cnt", color="orange")
    # plt.plot(heights, GEDs_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Avg. GED")
    # plt.title("Average of GED calculation: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.savefig("avg_GED_MUTAG.png")
    # plt.clf()
    # print("MUTAG - avg GED: Done")

    # # * Mutag - with baseline, using csv
    # mutag = main.load_graphs("MUTAG")
    # matrix_exact = pd.read_csv("MUTAG_Matrix_exact.csv", header=None).values
    # matrix_bgm_cnt = pd.read_csv("MUTAG_Matrix_bgm.csv", header=None).values
    # matrix_bgm_leon1 = pd.read_csv("MUTAG_Matrix_bgm_leon1.csv", header=None).values
    # matrix_bgm_leon2 = pd.read_csv("MUTAG_Matrix_bgm_leon2.csv", header=None).values
    # matrix_nx_0 = pd.read_csv("MUTAG_Matrix_nx_0.csv", header=None).values
    # # matrix_nx_1 = pd.read_csv("MUTAG_Matrix_nx_1.csv", header=None).values
    # # matrix_nx_2 = pd.read_csv("MUTAG_Matrix_nx_2.csv", header=None).values
    # errors_cnt = []
    # errors_bgm_cnt = []
    # errors_bgm_leon1 = []
    # errors_bgm_leon2 = []
    # errors_nx_0 = []
    # # errors_nx_1 = []
    # # errors_nx_2 = []
    # for height in heights:
    #     matrix_cnt = pd.read_csv(f"MUTAG_Matrix_cnt_{height}.csv", header=None).values
    #     error_cnt = 0
    #     error_bgm_cnt = 0
    #     error_bgm_leon1 = 0
    #     error_bgm_leon2 = 0
    #     error_nx_0 = 0
    #     # error_nx_1 = 0
    #     # error_nx_2 = 0
    #     for i in range(len(mutag)):
    #         for j in range(len(mutag)):
    #             if i == j:
    #                 continue
    #             error_cnt += abs(matrix_exact[i][j] - matrix_cnt[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_cnt[i][j]
    #             error_bgm_cnt += abs(matrix_exact[i][j] - matrix_bgm_cnt[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_bgm_cnt[i][j]
    #             error_bgm_leon1 += abs(matrix_exact[i][j] - matrix_bgm_leon1[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_bgm_leon1[i][j]
    #             error_bgm_leon2 += abs(matrix_exact[i][j] - matrix_bgm_leon2[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_bgm_leon2[i][j]
    #             error_nx_0 += abs(matrix_exact[i][j] - matrix_nx_0[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_nx_0[i][j]
    #             # error_nx_1 += abs(matrix_exact[i][j] - matrix_nx_1[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_nx_1[i][j]
    #             # error_nx_2 += abs(matrix_exact[i][j] - matrix_nx_2[i][j])/matrix_exact[i][j] if matrix_exact[i][j] != 0 else matrix_nx_2[i][j]
    #     error_cnt /= len(mutag) * len(mutag) - len(mutag)
    #     error_bgm_cnt /= len(mutag) * len(mutag) - len(mutag)
    #     error_bgm_leon1 /= len(mutag) * len(mutag) - len(mutag)
    #     error_bgm_leon2 /= len(mutag) * len(mutag) - len(mutag)
    #     error_nx_0 /= len(mutag) * len(mutag) - len(mutag)
    #     # error_nx_1 /= len(mutag) * len(mutag) - len(mutag)
    #     # error_nx_2 /= len(mutag) * len(mutag) - len(mutag)

    #     errors_cnt.append(error_cnt)
    #     errors_bgm_cnt.append(error_bgm_cnt)
    #     errors_bgm_leon1.append(error_bgm_leon1)
    #     errors_bgm_leon2.append(error_bgm_leon2)
    #     errors_nx_0.append(error_nx_0)
    #     # errors_nx_1.append(error_nx_1)
    #     # errors_nx_2.append(error_nx_2)
    
    # plt.plot(heights, errors_cnt, label="cnt", color="orange")
    # plt.plot(heights, errors_bgm_cnt, label="bgm_cnt", color="blue")
    # plt.plot(heights, errors_bgm_leon1, label="bgm_leon1", color="green")
    # plt.plot(heights, errors_bgm_leon2, label="bgm_leon2", color="red")
    # plt.plot(heights, errors_nx_0, label="nx_0", color="purple")
    # # plt.plot(heights, errors_nx_1, label="nx_1", color="brown")
    # # plt.plot(heights, errors_nx_2, label="nx_2", color="pink")
    # plt.xlabel("Height")
    # plt.ylabel("Avg. Error")
    # plt.title("Average of Approximation Error: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Adjust position
    # plt.tight_layout()
    # plt.savefig("avg_error_MUTAG.png")
    # plt.clf()
    # print("MUTAG - avg error: Done")

    # # * PTC_FM - single GED
    # cache = {}
    # ptc_fm = main.load_graphs("PTC_FM",101)
    # runtimes_cnt = []
    # runtimes_bgm = []
    # GEDs_cnt = []
    # GEDs_bgm = []

    # runtime_bgm = 0
    # GED_bgm = 0
    # for i in range(100):
    #     for j in range(100):
    #         basetime = t.time()
    #         GED_bgm += main.standard_bgm(ptc_fm[i+1], ptc_fm[j+1])[2]
    #         runtime_bgm += t.time() - basetime
    # for height in heights:
    #     runtimes_bgm.append(runtime_bgm / 10000)
    #     GEDs_bgm.append(GED_bgm / 10000)

    # for height in heights:
    #     print(f"Height: {height}")
    #     runtime_cnt = 0
    #     GED_cnt = 0
    #     for i in range(100):
    #         print(f"Run: {i}")
    #         for j in range(100):
    #             basetime = t.time()
    #             GED_cnt += main.calculate_GED_bgm(ptc_fm[i+1], ptc_fm[j+1], height=height, cache=cache)[2]
    #             runtime_cnt += t.time() - basetime
    #     runtimes_cnt.append(runtime_cnt / 10000)
    #     GEDs_cnt.append(GED_cnt / 10000)

    # plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    # plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title("Runtime of GED calculation: PTC_FM")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.savefig("runtime_ged_PTC_FM.png")
    # plt.clf()
    # print("PTC_FM - single GED: Done")

    # plt.plot(heights, GEDs_cnt, label="cnt", color="orange")
    # plt.plot(heights, GEDs_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Avg. GED")
    # plt.title("Average of GED calculation: PTC_FM")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.savefig("avg_GED_PTC_FM.png")
    # plt.clf()
    # print("PTC_FM - avg GED: Done")


    # ? Graph-classification

    # # * MUTAG - 90 Train, 90 Test
    # mutag = main.load_graphs("MUTAG")
    # accuracies_cnt = []
    # accuracies_bgm = []

    # m = 20

    # for height in heights:
    #     accuracy_bgm = 0
    #     for i in range(m):
    #         accuracy_bgm += train_kNN(mutag, 90, 90, method="bgm")
    #     accuracies_bgm.append(accuracy_bgm / m)
    
    # for height in heights:
    #     print(f"Height: {height}")
    #     accuracy_cnt = 0
    #     for i in range(m):
    #         print(f"Run: {i}")
    #         accuracy_cnt += train_kNN(mutag, 90, 90, height=height, method="cnt")
    #     accuracies_cnt.append(accuracy_cnt / m)

    # plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    # plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy of k-NN classification: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.savefig("accuracy_kNN_MUTAG.png")
    # plt.clf()
    # print("MUTAG - 90 Train, 90 Test: Done")

    # # * 1 nn classification

    # mutag = main.load_graphs("MUTAG")
    # tp, fp, tn, fn, time_took = test_knn(mutag, method="cnt", height=5, k_param=0)
    # print("MUTAG")
    # print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, Time: {time_took}")

    # print("------------------------------------")

    # aids = main.load_graphs("AIDS")
    # tp, fp, tn, fn = test_knn(aids, method="cnt", height=5, k_param=0)
    # print("AIDS")
    # print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    # * 3nn classification using matrix
    mutag = main.load_graphs("MUTAG")

    accuracies_exact = []
    accuracies_cnt = []
    accuracies_bgm = []

    for height in heights:
        mutag_matrix = pd.read_csv("MUTAG_Matrix_exact.csv", header=None).values
        accuracy_exact = knn_matrix(mutag, mutag_matrix, test_size=0.2)
        accuracies_exact.append(accuracy_exact)
        
        mutag_matrix = pd.read_csv(f"MUTAG_Matrix_cnt_{height}.csv", header=None).values
        accuracy_cnt = knn_matrix(mutag, mutag_matrix, test_size=0.2)
        accuracies_cnt.append(accuracy_cnt)
        
        mutag_matrix = pd.read_csv("MUTAG_Matrix_bgm.csv", header=None).values
        accuracy_bgm = knn_matrix(mutag, mutag_matrix, test_size=0.2)
        accuracies_bgm.append(accuracy_bgm)

    plt.plot(heights, accuracies_exact, label="exact", color="green")
    plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of 3-NN classification: MUTAG")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("accuracy_3NN_matrix_MUTAG.png")

    plt.clf()

    # * 1nn classification using matrix
    mutag = main.load_graphs("MUTAG")

    accuracies_exact = []
    accuracies_cnt = []
    accuracies_bgm = []

    for height in heights:
        mutag_matrix = pd.read_csv("MUTAG_Matrix_exact.csv", header=None).values
        accuracy_exact = knn_matrix(mutag, mutag_matrix, test_size=0.2, n_neighbors=1)
        accuracies_exact.append(accuracy_exact)
        
        mutag_matrix = pd.read_csv(f"MUTAG_Matrix_cnt_{height}.csv", header=None).values
        accuracy_cnt = knn_matrix(mutag, mutag_matrix, test_size=0.2, n_neighbors=1)
        accuracies_cnt.append(accuracy_cnt)
        
        mutag_matrix = pd.read_csv("MUTAG_Matrix_bgm.csv", header=None).values
        accuracy_bgm = knn_matrix(mutag, mutag_matrix, test_size=0.2, n_neighbors=1)
        accuracies_bgm.append(accuracy_bgm)

    plt.plot(heights, accuracies_exact, label="exact", color="green")
    plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of 1-NN classification: MUTAG")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("accuracy_1NN_matrix_MUTAG.png")

    plt.clf()

    # # ? Graph-clustering

    # # * MUTAG - 90 - Clustering
    # mutag_90 = main.load_graphs("MUTAG",90)
    # accuracies_cnt = []
    # accuracies_bgm = []

    # m = 20

    # for height in heights:
    #     accuracy_bgm = 0
    #     for i in range(m):
    #         accuracy_bgm += cluster_graphs(mutag_90, method="bgm")[1]
    #     accuracies_bgm.append(accuracy_bgm / m)
    
    # for height in heights:
    #     print(f"Height: {height}")
    #     accuracy_cnt = 0
    #     for i in range(m):
    #         print(f"Run: {i}")
    #         accuracy_cnt += cluster_graphs(mutag_90, height=height, method="cnt")[1]
    #     accuracies_cnt.append(accuracy_cnt / m)

    # plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    # plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy of clustering: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.savefig("accuracy_clustering_MUTAG.png")
    # plt.clf()
    # print("MUTAG - 90 - Clustering: Done")

    # # * spectral clustering
    # mutag = main.load_graphs("MUTAG")

    # accuracies_exact = []
    # accuracies_cnt = []
    # accuracies_bgm = []

    # for height in heights:
    #     mutag_matrix = pd.read_csv("MUTAG_Matrix_exact.csv", header=None).values
    #     accuracy_exact = spectral_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_exact.append(accuracy_exact)
        
    #     mutag_matrix = pd.read_csv(f"MUTAG_Matrix_cnt_{height}.csv", header=None).values
    #     accuracy_cnt = spectral_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_cnt.append(accuracy_cnt)
        
    #     mutag_matrix = pd.read_csv("MUTAG_Matrix_bgm.csv", header=None).values
    #     accuracy_bgm = spectral_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_bgm.append(accuracy_bgm)

    # plt.plot(heights, accuracies_exact, label="exact", color="green")
    # plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    # plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy of spectral clustering: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Adjust position
    # plt.tight_layout()
    # plt.savefig("accuracy_spectral_clustering_MUTAG.png")
    # plt.clf()

    # print("MUTAG - spectral clustering: Done")

    # # * agglomerative clustering

    # mutag = main.load_graphs("MUTAG")

    # accuracies_exact = []
    # accuracies_cnt = []
    # accuracies_bgm = []

    # for height in heights:
    #     mutag_matrix = pd.read_csv("MUTAG_Matrix_exact.csv", header=None).values
    #     accuracy_exact = agglomerative_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_exact.append(accuracy_exact)
        
    #     mutag_matrix = pd.read_csv(f"MUTAG_Matrix_cnt_{height}.csv", header=None).values
    #     accuracy_cnt = agglomerative_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_cnt.append(accuracy_cnt)
        
    #     mutag_matrix = pd.read_csv("MUTAG_Matrix_bgm.csv", header=None).values
    #     accuracy_bgm = agglomerative_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_bgm.append(accuracy_bgm)

    # plt.plot(heights, accuracies_exact, label="exact", color="green")
    # plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    # plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy of agglomerative clustering: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Adjust position
    # plt.tight_layout()
    # plt.savefig("accuracy_agglomerative_clustering_MUTAG.png")
    # plt.clf()

    # print("MUTAG - agglomerative clustering: Done")

    # # * k-metoid clustering

    # mutag = main.load_graphs("MUTAG")

    # accuracies_exact = []
    # accuracies_cnt = []
    # accuracies_bgm = []

    # for height in heights:
    #     mutag_matrix = pd.read_csv("MUTAG_Matrix_exact.csv", header=None).values
    #     accuracy_exact = k_metoid_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_exact.append(accuracy_exact)
        
    #     mutag_matrix = pd.read_csv(f"MUTAG_Matrix_cnt_{height}.csv", header=None).values
    #     accuracy_cnt = k_metoid_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_cnt.append(accuracy_cnt)
        
    #     mutag_matrix = pd.read_csv("MUTAG_Matrix_bgm.csv", header=None).values
    #     accuracy_bgm = k_metoid_clustering(mutag, mutag_matrix, n_clusters=2)
    #     accuracies_bgm.append(accuracy_bgm)

    # plt.plot(heights, accuracies_exact, label="exact", color="green")
    # plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    # plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy of k-metoid clustering: MUTAG")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Adjust position
    # plt.tight_layout()
    # plt.savefig("accuracy_k_metoid_clustering_MUTAG.png")
    # plt.clf()

    # print("MUTAG - k-metoid clustering: Done")