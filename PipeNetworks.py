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

    def __init__(self, nodes_file_dir, tubes_file_dir):
        nodes_input = np.genfromtxt(nodes_file_dir, delimiter=',')
        tubes_input = np.genfromtxt(tubes_file_dir, delimiter=',')
        self.nodes = []
        self.tubes = []
        for i in range(len(nodes_input[:, 0])):
            coordinates = [nodes_input[i, 0], nodes_input[i, 1], nodes_input[i, 2]]
            self.nodes.append(Node(coordinates))
        for i in range(len(tubes_input[:, 0])):
            self.tubes.append(Tube(tubes_input[i, 0:2]))
            self.tubes[i].length = tubes_input[i, 2]

    # given a set of nodes and tubes,
    # it finds the specific tube that connects to each node
    def FindConnectedTubes(self):
        for i in range(Tube.number_of_tubes):
            self.nodes[self.tubes[i].connectedNodes[0]].connectedTubes.append(i)
            self.nodes[self.tubes[i].connectedNodes[1]].connectedTubes.append(i)

    def FindNeighbouringElements(self):
        for i in range(Node.number_of_nodes):
            for tube_index in self.nodes[i].connectedTubes:
                if self.tubes[tube_index].connectedNodes[0] != i:
                    self.nodes[i].neighbouringNodes.append(self.tubes[tube_index].connectedNodes[0])
                else:
                    self.nodes[i].neighbouringNodes.append(self.tubes[tube_index].connectedNodes[1])

    def Plot(self):
        plt.figure()
        if self.nodes[0].z is not None:
            plt.axes(projection='3d')
        for i in range(Tube.number_of_tubes):
            xs = [self.nodes[self.tubes[i].connectedNodes[0]].x, self.nodes[self.tubes[i].connectedNodes[1]].x]
            ys = [self.nodes[self.tubes[i].connectedNodes[0]].y, self.nodes[self.tubes[i].connectedNodes[1]].y]
            if self.nodes[0].z is not None:
                zs = [self.nodes[self.tubes[i].connectedNodes[0]].z, self.nodes[self.tubes[i].connectedNodes[1]].z]
                plt.plot(xs, ys, zs, 'ko-')
            else:
                plt.plot(xs, ys, 'ko-')

        plt.grid()
        plt.show()











