import matplotlib.pyplot as plt
import numpy as np

class Node:
    number_of_nodes = 0

    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]
        if len(coordinates) == 3:
            self.z = coordinates[2]
        else:
            self.z = None
        self.p0 = None
        self.connectedTubes = []
        self.neighbouringNodes = []
        Node.number_of_nodes += 1


class Tube:
    number_of_tubes = 0

    def __init__(self, connected_nodes):
        self.length = -1
        self.k = None
        self.Re = None
        self.Q = None
        self.lamda = None
        self.connectedNodes = list(connected_nodes.astype(int))  # index of nodes at each end
        Tube.number_of_tubes += 1


class Network:

    @staticmethod
    def Create(nodes_file_dir, tubes_file_dir):
        nodes_input = np.genfromtxt(nodes_file_dir, delimiter=',')
        tubes_input = np.genfromtxt(tubes_file_dir, delimiter=',')
        nodes_temp = []
        tubes_temp = []
        for i in range(len(nodes_input[:, 0])):
            coordinates = [nodes_input[i, 0], nodes_input[i, 1], nodes_input[i, 2]]
            nodes_temp.append(Node(coordinates))
        for i in range(len(tubes_input[:, 0])):
            tubes_temp.append(Tube(tubes_input[i, 0:2]))
            tubes_temp[i].length = tubes_input[i, 2]
        return nodes_temp, tubes_temp

    # given a set of nodes and tubes,
    # it finds the specific tube that connects to each node
    @staticmethod
    def FindConnectedTubes(nodes: list[Node], tubes: list[Tube]):
        for i in range(Tube.number_of_tubes):
            nodes[tubes[i].connectedNodes[0]].connectedTubes.append(i)
            nodes[tubes[i].connectedNodes[1]].connectedTubes.append(i)

    @staticmethod
    def FindNeighbouringElements(nodes: list[Node], tubes: list[Tube]):
        for i in range(Node.number_of_nodes):
            for tube_index in nodes[i].connectedTubes:
                if tubes[tube_index].connectedNodes[0] != i:
                    nodes[i].neighbouringNodes.append(tubes[tube_index].connectedNodes[0])
                else:
                    nodes[i].neighbouringNodes.append(tubes[tube_index].connectedNodes[1])

    @staticmethod
    def Plot(nodes: list[Node], tubes: list[Tube]):
        plt.figure()
        if nodes[0].z is not None:
            plt.axes(projection='3d')
        for i in range(Tube.number_of_tubes):
            xs = [nodes[tubes[i].connectedNodes[0]].x, nodes[tubes[i].connectedNodes[1]].x]
            ys = [nodes[tubes[i].connectedNodes[0]].y, nodes[tubes[i].connectedNodes[1]].y]
            if nodes[0].z is not None:
                zs = [nodes[tubes[i].connectedNodes[0]].z, nodes[tubes[i].connectedNodes[1]].z]
                plt.plot(xs, ys, zs, 'ko-')
            else:
                plt.plot(xs, ys, 'ko-')

        plt.grid()
        plt.show()











