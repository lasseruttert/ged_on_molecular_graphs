import networkx as nx
import matplotlib.pyplot as plt
import time as t
import main as main

if __name__ == "__main__":

    dataset_name = "MUTAG"
    graphs = main.load_graphs(dataset_name)

    # total_time = 0
    # for i in range(10):
    #     basetime = t.time()
    #     main.calculate_GED_bgm(graphs[1], graphs[2])
    #     total_time += t.time() - basetime

    # print(f"Average time: {total_time / 10}s")

    # heights = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    # runtimes_cnt = []
    # runtimes_bgm = []
    # for height in heights:
    #     total_time_cnt = 0
    #     total_time_bgm = 0
    #     for i in range(30):
    #         x = r.randint(1, 188)
    #         y = r.randint(1, 188)
    #         basetime = t.time()
    #         main.calculate_GED_bgm(graphs[x], graphs[y], height=height)
    #         total_time_cnt += t.time() - basetime
    #         basetime = t.time()
    #         main.standard_bgm(graphs[x], graphs[y])
    #         total_time_bgm += t.time() - basetime
    #     runtimes_cnt.append(total_time_cnt / 30)
    #     runtimes_bgm.append(total_time_bgm / 30)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, runtimes_cnt, label="cnt", color="orange")
    # plt.plot(heights, runtimes_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title(f"Runtime of GED calculation: {dataset_name}")

    # plt.savefig(f"runtime_ged_{dataset_name}.png") 

    # heights = [1,2,3,4,5,6,7,8,9,10]
    # runtimes = []
    # for height in heights:
    #     total_time = 0
    #     for i in range(10):
    #         basetime = t.time()
    #         main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:5]}, height=height, k=0)
    #         total_time += t.time() - basetime
    #     runtimes.append(total_time / 10)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, runtimes)
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title(f"Runtime of cost matrix calculation: {dataset_name}")

    # plt.savefig(f"runtime_cost_matrix_{dataset_name}.png")

    # avg error
    # heights = [1,2,3,4,5,6,7,8,9,10]
    # error_cnt = []
    # error_bgm = []
    # actual = 10
    # for height in heights:
    #     print(f"Height: {height}")
    #     current_error = 0
    #     for i in range(100):
    #         _,_,calculated,_,_ = main.calculate_GED_bgm(graphs[1], graphs[2], height=height)
    #         current_error += (abs(calculated - actual)/actual)
    #     error_cnt.append(current_error / 100)
    #     current_error = 0
    #     for i in range(100):
    #         _,_,calculated,_,_ = main.standard_bgm(graphs[1], graphs[2])
    #         current_error += (abs(calculated - actual)/actual)
    #     error_bgm.append(current_error / 100)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, error_cnt, label="cnt", color="orange")
    # plt.plot(heights, error_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Error")
    # plt.title(f"Error of GED calculation: {dataset_name}")

    # plt.savefig(f"error_ged_{dataset_name}.png")

    # heights = [1,2,3,4,5,6,7,8,9,10]
    # errors_cnt = []
    # errors_bgm = []
    # x = 1
    # y = 13
    # actual = nx.graph_edit_distance(graphs[x],graphs[y],node_match=main.node_match,edge_match=main.edge_match, timeout=10)
    # for height in heights:
    #     print(f"Height: {height}")
    #     error_cnt = 0
    #     error_bgm = 0
    #     for i in range(100):
    #         print(i)
    #         # x = r.randint(1, 188)
    #         # y = r.randint(1, 188)
    #         # actual = nx.graph_edit_distance(graphs[x],graphs[y],node_match=main.node_match,edge_match=main.edge_match, timeout=0.5)
    #         _,_,cnt,_,_ = main.calculate_GED_bgm(graphs[x], graphs[y], height=height)
    #         error_cnt += (cnt - actual)/actual if actual != 0 else 0
    #         _,_,bgm,_,_ = main.standard_bgm(graphs[x], graphs[y])
    #         error_bgm += (bgm - actual)/actual if actual != 0 else 0
    #     errors_cnt.append(error_cnt / 100)
    #     errors_bgm.append(error_bgm / 100)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, errors_cnt, label="cnt", color="orange")
    # plt.plot(heights, errors_bgm, label="bgm", color="blue")
    # plt.xlabel("Height")
    # plt.ylabel("Error")
    # plt.title(f"Error of GED (CNT vs BGM): {dataset_name}")

    # plt.savefig(f"error_ged_cnt_vs_bgm_{dataset_name}.png")


    heights = [1,2,3,4,5,6,7,8,9,10]
    GEDs_cnt = []
    GEDs_bgm = []
    GEDs_diff = []
    for height in heights:
        print(f"Height: {height}")
        GED_cnt = 0
        GED_bgm = 0
        for i in range(100):
            for j in range(100):
                print(height,i,j)
                _,_,cnt,_,_ = main.calculate_GED_bgm(graphs[i+1], graphs[j+1], height=height)
                _,_,bgm,_,_ = main.standard_bgm(graphs[i+1], graphs[j+1])
                GED_cnt += cnt
                GED_bgm += bgm
        GEDs_cnt.append(GED_cnt / 10000)
        GEDs_bgm.append(GED_bgm / 10000)
        GEDs_diff.append(abs(GED_cnt - GED_bgm)/10000)

    # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    plt.plot(heights, GEDs_cnt, label="cnt", color="orange")
    plt.plot(heights, GEDs_bgm, label="bgm", color="blue")
    plt.plot(heights, GEDs_diff, label="diff", color="green")
    plt.xlabel("Height")
    plt.ylabel("GED")
    plt.title(f"GED of CNT vs BGM (with diff): {dataset_name}")

    plt.savefig(f"ged_cnt_vs_bgm_diff_{dataset_name}.png")

    print("Done")