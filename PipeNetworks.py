import matplotlib.pyplot as plt
import numpy as np


class Node:

    def __init__(self, coordinates):
        self.index = None
        self.x = coordinates[0]
        self.y = coordinates[1]
        if len(coordinates) == 3:
            self.z = coordinates[2]
        else:
            self.z = None
        self.p_static = None  # pressure [Pa]
        self.p0 = None  # pressure [Pa]
        self.consumption = 0  # Consumption (+: outbound, -: inbound)
        self.zeta = 0
        self.connectedTubes = []
        self.neighbouringNodes = []


class Tube:

    def __init__(self, connected_nodes):
        self.index = None
        self.length = None
        self.diameter = None
        self.Re = None
        self.Q = None
        self.kappa = None
        self.lamda = None
        self.zeta = 0
        self.connectedNodes = list(connected_nodes.astype(int))  # index of nodes at each end


class Network:

    def __init__(self, nodes_file_dir, tubes_file_dir):
        nodes_input = np.genfromtxt(nodes_file_dir, delimiter=',')
        tubes_input = np.genfromtxt(tubes_file_dir, delimiter=',')
        self.nodes = []
        self.tubes = []
        for i in range(len(nodes_input[:, 0])):
            coordinates = [nodes_input[i, 0], nodes_input[i, 1], nodes_input[i, 2]]
            self.nodes.append(Node(coordinates))
            self.nodes[i].zeta = nodes_input[i, 3]
            self.nodes[i].consumption = nodes_input[i, 4] / 3600  # [m^3/h -> m^3/s]
            self.nodes[i].index = i
        for i in range(len(tubes_input[:, 0])):
            self.tubes.append(Tube(tubes_input[i, 0:2]))
            self.tubes[i].length = tubes_input[i, 2]
            self.tubes[i].zeta = tubes_input[i, 3]
            self.tubes[i].index = i
        self.number_of_nodes = len(self.nodes)
        self.number_of_tubes = len(self.tubes)

    # given a set of nodes and tubes,
    # it finds the specific tube that connects to each node
    def FindConnectedTubes(self):
        for i in range(self.number_of_tubes):
            self.nodes[self.tubes[i].connectedNodes[0]].connectedTubes.append(i)
            self.nodes[self.tubes[i].connectedNodes[1]].connectedTubes.append(i)

    def FindNeighbouringElements(self):
        for i in range(self.number_of_nodes):
            for tube_index in self.nodes[i].connectedTubes:
                if self.tubes[tube_index].connectedNodes[0] != i:
                    self.nodes[i].neighbouringNodes.append(self.tubes[tube_index].connectedNodes[0])
                else:
                    self.nodes[i].neighbouringNodes.append(self.tubes[tube_index].connectedNodes[1])

    def GetTubeIndexConnectingIwithJ(self, node_i_index: int, node_j_index: int):
        for conn_tube_index in self.nodes[node_i_index].connectedTubes:
            if self.tubes[conn_tube_index].connectedNodes[0] == node_j_index \
                    or self.tubes[conn_tube_index].connectedNodes[1] == node_j_index:
                return conn_tube_index

    def Plot(self):
        plt.figure()
        if self.nodes[0].z is not None:
            plt.axes(projection='3d')
        for i in range(self.number_of_tubes):
            xs = [self.nodes[self.tubes[i].connectedNodes[0]].x, self.nodes[self.tubes[i].connectedNodes[1]].x]
            ys = [self.nodes[self.tubes[i].connectedNodes[0]].y, self.nodes[self.tubes[i].connectedNodes[1]].y]
            if self.nodes[0].z is not None:
                zs = [self.nodes[self.tubes[i].connectedNodes[0]].z, self.nodes[self.tubes[i].connectedNodes[1]].z]
                plt.plot(xs, ys, zs, 'ko-')
            else:
                plt.plot(xs, ys, 'ko-')

        plt.grid()
        plt.show()


def sign(x, y):
    if x >= y:
        return 1
    else:
        return -1
