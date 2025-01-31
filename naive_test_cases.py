import unittest
import main as main
import networkx as nx
import numpy as np
import numpy.testing as npt

class TestMain(unittest.TestCase):
    def test_1(self):
        # Test two identical graphs, which should return a 0 cost matrix
        
        graph1 = nx.Graph()
        graph1.graph["id"] = 1
        graph1.add_node(1, label="A")
        graph1.add_node(2, label="B")
        graph1.add_node(3, label="C")
        graph1.add_node(4, label="D")

        graph1.add_edge(1, 2, label="a")
        graph1.add_edge(2, 3, label="b")
        graph1.add_edge(3, 4, label="c")

        graph2 = nx.Graph()
        graph2.graph["id"] = 2
        graph2.add_node(1, label="A")
        graph2.add_node(2, label="B")
        graph2.add_node(3, label="C")
        graph2.add_node(4, label="D")

        graph2.add_edge(1, 2, label="a")
        graph2.add_edge(2, 3, label="b")
        graph2.add_edge(3, 4, label="c")

        test_graphs = {}

        test_graphs[1] = graph1
        test_graphs[2] = graph2

        expected_output = np.zeros((2, 2))
        test_output = main.calculate_cost_matrix(test_graphs)

        npt.assert_array_equal(expected_output, test_output)

    def test_2(self):
        # two graphs which only have a different root label

        graph1 = nx.Graph()
        graph1.graph["id"] = 1
        graph1.add_node(1, label="A")
        graph1.add_node(2, label="B")
        graph1.add_node(3, label="C")

        graph1.add_edge(1, 2, label="a")
        graph1.add_edge(2, 3, label="b")

        graph2 = nx.Graph()
        graph2.graph["id"] = 2
        graph2.add_node(1, label="X")
        graph2.add_node(2, label="B")
        graph2.add_node(3, label="C")

        graph2.add_edge(1, 2, label="a")
        graph2.add_edge(2, 3, label="b")

        test_graphs = {}

        test_graphs[1] = graph1
        test_graphs[2] = graph2

        expected_output = np.matrix([[0, 1], [1, 0]])
        test_output = main.calculate_cost_matrix(test_graphs)

        npt.assert_array_equal(expected_output, test_output)

    def test_3(self):
        # two graphs which only have a different edge label

        graph1 = nx.Graph()
        graph1.graph["id"] = 1
        graph1.add_node(1, label="A")
        graph1.add_node(2, label="B")
        graph1.add_node(3, label="C")

        graph1.add_edge(1, 2, label="a")
        graph1.add_edge(2, 3, label="b")

        graph2 = nx.Graph()
        graph2.graph["id"] = 2
        graph2.add_node(1, label="A")
        graph2.add_node(2, label="B")
        graph2.add_node(3, label="C")

        graph2.add_edge(1, 2, label="a")
        graph2.add_edge(2, 3, label="c")

        test_graphs = {}

        test_graphs[1] = graph1
        test_graphs[2] = graph2

        expected_output = np.matrix([[0, 1], [1, 0]])
        test_output = main.calculate_cost_matrix(test_graphs)

        npt.assert_array_equal(expected_output, test_output)
    
    def test_4(self):
        # two graphs: one with 1 node, one with 4 nodes

        graph1 = nx.Graph()
        graph1.graph["id"] = 1
        graph1.add_node(1, label="A")

        graph2 = nx.Graph()
        graph2.graph["id"] = 2
        graph2.add_node(1, label="A")
        graph2.add_node(2, label="B")
        graph2.add_node(3, label="C")
        graph2.add_node(4, label="D")

        graph2.add_edge(1, 2, label="a")
        graph2.add_edge(1, 3, label="b")
        graph2.add_edge(1, 4, label="c")

        test_graphs = {}

        test_graphs[1] = graph1
        test_graphs[2] = graph2

        expected_output = np.matrix([[0, 6], [6, 0]])
        test_output = main.calculate_cost_matrix(test_graphs)

        npt.assert_array_equal(expected_output, test_output)

    def test_5(self):
        # two graphs: one with 1 node, one with 3 nodes in a line

        graph1 = nx.Graph()
        graph1.graph["id"] = 1
        graph1.add_node(1, label="A")

        graph2 = nx.Graph()
        graph2.graph["id"] = 2
        graph2.add_node(1, label="A")
        graph2.add_node(2, label="B")
        graph2.add_node(3, label="C")

        graph2.add_edge(1, 2, label="a")
        graph2.add_edge(2, 3, label="b")

        test_graphs = {}

        test_graphs[1] = graph1
        test_graphs[2] = graph2

        expected_output = np.matrix([[0, 4], [4, 0]])
        test_output = main.calculate_cost_matrix(test_graphs)

        npt.assert_array_equal(expected_output, test_output)


if __name__ == '__main__':
    unittest.main()

