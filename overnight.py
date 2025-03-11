import main as main
import time as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from graph_classification import train_kNN
from graph_clustering import cluster_graphs

if __name__ == "__main__":

    heights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # ? Costmatrix

    # * MUTAG - Full - CNT
    mutag_full = main.load_graphs("MUTAG")
    mutag_full_matrix,_,_ = main.calculate_cost_matrix(mutag_full, height=5)
    np.savetxt(f"MUTAG_full_cost_matrix.csv", mutag_full_matrix, delimiter=",",fmt="%d")
    plt.figure(figsize=(10, 8))
    sns.heatmap(mutag_full_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
    plt.xlabel('Graphs')
    plt.ylabel('Graphs')
    plt.title("MUTAG - Full - CNT")
    plt.savefig("MUTAG_full_cost_matrix.png")
    plt.clf()
    print("MUTAG - Full - CNT: Done")

    # * MUTAG - 20 - CNT
    mutag_20 = main.load_graphs("MUTAG", 20)
    mutag_20_cnt,_,_ = main.calculate_cost_matrix(mutag_20, height=5)
    np.savetxt(f"MUTAG_20_cost_matrix.csv", mutag_20_cnt, delimiter=",",fmt="%d")
    plt.figure(figsize=(10, 8))
    sns.heatmap(mutag_20_cnt, annot=True, fmt="d", cmap="plasma", cbar=True)
    plt.xlabel('Graphs')
    plt.ylabel('Graphs')
    plt.title("MUTAG - 20 - CNT")
    plt.savefig("MUTAG_20_cost_matrix.png")
    plt.clf()
    print("MUTAG - 20 - CNT: Done")

    # * MUTAG - 20 - BGM
    mutag_20 = main.load_graphs("MUTAG", 20)
    mutag_20_bgm,_,_ = main.standard_bgm_matrix(mutag_20)
    np.savetxt(f"MUTAG_20_bgm_cost_matrix.csv", mutag_20_bgm, delimiter=",",fmt="%d")
    plt.figure(figsize=(10, 8))
    sns.heatmap(mutag_20_bgm, annot=True, fmt="d", cmap="plasma", cbar=True)
    plt.xlabel('Graphs')
    plt.ylabel('Graphs')
    plt.title("MUTAG - 20 - BGM")
    plt.savefig("MUTAG_20_bgm_cost_matrix.png")
    plt.clf()
    print("MUTAG - 20 - BGM: Done")

    # * MUTAG - 20 - Diff
    mutag_20_diff = mutag_20_cnt - mutag_20_bgm
    np.savetxt(f"MUTAG_20_diff_cost_matrix.csv", mutag_20_diff, delimiter=",",fmt="%d")
    plt.figure(figsize=(10, 8))
    sns.heatmap(mutag_20_diff, annot=True, fmt="d", cmap="PuOr", cbar=True)
    plt.xlabel('Graphs')
    plt.ylabel('Graphs')
    plt.title("MUTAG - 20 - Diff")
    plt.savefig("MUTAG_20_diff_cost_matrix.png")
    plt.clf()
    print("MUTAG - 20 - Diff: Done")


    # ? Runtime + Precision

    # * MUTAG - single GED
    cache = {}
    mutag_101 = main.load_graphs("MUTAG", 101)
    runtimes_cnt = []
    runtimes_bgm = []
    GEDs_cnt = []
    GEDs_bgm = []

    runtime_bgm = 0
    GED_bgm = 0
    for i in range(100):
        for j in range(100):
            basetime = t.time()
            GED_bgm += main.standard_bgm(mutag_101[i+1], mutag_101[j+1])[2]
            runtime_bgm += t.time() - basetime
    for height in heights:
        runtimes_bgm.append(runtime_bgm / 10000)
        GEDs_bgm.append(GED_bgm / 10000)

    for height in heights:
        print(f"Height: {height}")
        runtime_cnt = 0
        GED_cnt = 0
        for i in range(100):
            print(f"Run: {i}")
            for j in range(100):
                basetime = t.time()
                GED_cnt += main.calculate_GED_bgm(mutag_101[i+1], mutag_101[j+1], height=height, cache=cache)[2]
                runtime_cnt += t.time() - basetime
        runtimes_cnt.append(runtime_cnt / 10000)
        GEDs_cnt.append(GED_cnt / 10000)

    plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Runtime")
    plt.title("Runtime of GED calculation: MUTAG")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("runtime_ged_MUTAG.png")
    plt.clf()
    print("MUTAG - single GED: Done")

    plt.plot(heights, GEDs_cnt, label="cnt", color="orange")
    plt.plot(heights, GEDs_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Avg. GED")
    plt.title("Average of GED calculation: MUTAG")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("avg_GED_MUTAG.png")
    plt.clf()
    print("MUTAG - avg GED: Done")

    # * PTC_FM - single GED
    cache = {}
    ptc_fm = main.load_graphs("PTC_FM",101)
    runtimes_cnt = []
    runtimes_bgm = []
    GEDs_cnt = []
    GEDs_bgm = []

    runtime_bgm = 0
    GED_bgm = 0
    for i in range(100):
        for j in range(100):
            basetime = t.time()
            GED_bgm += main.standard_bgm(ptc_fm[i+1], ptc_fm[j+1])[2]
            runtime_bgm += t.time() - basetime
    for height in heights:
        runtimes_bgm.append(runtime_bgm / 10000)
        GEDs_bgm.append(GED_bgm / 10000)

    for height in heights:
        print(f"Height: {height}")
        runtime_cnt = 0
        GED_cnt = 0
        for i in range(100):
            print(f"Run: {i}")
            for j in range(100):
                basetime = t.time()
                GED_cnt += main.calculate_GED_bgm(ptc_fm[i+1], ptc_fm[j+1], height=height, cache=cache)[2]
                runtime_cnt += t.time() - basetime
        runtimes_cnt.append(runtime_cnt / 10000)
        GEDs_cnt.append(GED_cnt / 10000)

    plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Runtime")
    plt.title("Runtime of GED calculation: PTC_FM")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("runtime_ged_PTC_FM.png")
    plt.clf()
    print("PTC_FM - single GED: Done")

    plt.plot(heights, GEDs_cnt, label="cnt", color="orange")
    plt.plot(heights, GEDs_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Avg. GED")
    plt.title("Average of GED calculation: PTC_FM")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("avg_GED_PTC_FM.png")
    plt.clf()
    print("PTC_FM - avg GED: Done")

    # * ENZYMES - single GED
    # cache = {}
    # enzymes = main.load_graphs("ENZYMES",101)
    # runtimes_cnt = []
    # runtimes_bgm = []

    # for height in heights:
    #     runtime_cnt = 0
    #     runtime_bgm = 0
    #     for i in range(100):
    #         for j in range(100):
    #             basetime = t.time()
    #             main.calculate_GED_bgm(enzymes[i+1], enzymes[j+1], height=height, cache=cache)
    #             runtime_cnt += t.time() - basetime
    #             basetime = t.time()
    #             main.standard_bgm(enzymes[i+1], enzymes[j+1])
    #             runtime_bgm += t.time() - basetime
    #     runtimes_cnt.append(runtime_cnt / 10000)
    #     runtimes_bgm.append(runtime_bgm / 10000)

    # plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    # plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title("Runtime of GED calculation: ENZYMES")
    # plt.legend()
    # plt.savefig("runtime_ged_ENZYMES.png")
    # plt.clf()
    # print("ENZYMES - single GED: Done")

    # * MUTAG - 20 x 20 Matrix
    mutag_20 = main.load_graphs("MUTAG", 20)
    runtimes_cnt = []
    runtimes_bgm = []

    for height in heights:
        runtime_cnt = 0
        runtime_bgm = 0
        for i in range(10):
                basetime = t.time()
                main.calculate_cost_matrix(mutag_20, height=height)
                runtime_cnt += t.time() - basetime
                basetime = t.time()
                main.standard_bgm_matrix(mutag_20)
                runtime_bgm += t.time() - basetime
        runtimes_cnt.append(runtime_cnt / 10)
        runtimes_bgm.append(runtime_bgm / 10)

    plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Runtime")
    plt.title("Runtime of cost matrix calculation: MUTAG")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("runtime_cost_matrix_MUTAG.png")    
    plt.clf()
    print("MUTAG - 20 x 20 Matrix: Done")


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


    # ? Graph-clustering

    # * MUTAG - 90 - Clustering
    mutag_90 = main.load_graphs("MUTAG",90)
    accuracies_cnt = []
    accuracies_bgm = []

    m = 20

    for height in heights:
        accuracy_bgm = 0
        for i in range(m):
            accuracy_bgm += cluster_graphs(mutag_90, method="bgm")[1]
        accuracies_bgm.append(accuracy_bgm / m)
    
    for height in heights:
        print(f"Height: {height}")
        accuracy_cnt = 0
        for i in range(m):
            print(f"Run: {i}")
            accuracy_cnt += cluster_graphs(mutag_90, height=height, method="cnt")[1]
        accuracies_cnt.append(accuracy_cnt / m)

    plt.plot(heights, accuracies_cnt, label="cnt", color="orange")
    plt.plot(heights, accuracies_bgm, label="bgm", color="blue")
    plt.xlabel("Height")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of clustering: MUTAG")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("accuracy_clustering_MUTAG.png")
    plt.clf()
    print("MUTAG - 90 - Clustering: Done")