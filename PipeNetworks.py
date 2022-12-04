from math import *
import matplotlib.pyplot as plt
from ArrowPlot3d import *


class Node:

    def __init__(self, coordinates):
        self.index = None
        self.x = coordinates[0]
        self.y = coordinates[1]
        if len(coordinates) == 3:
            self.z = coordinates[2]
        else:
            self.z = 0
        self.p_static = None  # pressure [Pa]
        self.p0 = None  # pressure [Pa]
        self.consumption = 0  # Consumption (+: inbound, -: outbound)
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

    def __init__(self, nodes_file_dir: str, tubes_file_dir: str):
        nodes_input = np.genfromtxt(nodes_file_dir, delimiter=',', skip_header=1)
        tubes_input = np.genfromtxt(tubes_file_dir, delimiter=',', skip_header=1)
        self.nodes = []
        self.tubes = []
        # Reading Nodes from File
        for i in range(len(nodes_input[:, 0])):
            coordinates = [nodes_input[i, 0], nodes_input[i, 1], nodes_input[i, 2]]
            self.nodes.append(Node(coordinates))
            self.nodes[i].zeta = nodes_input[i, 3]
            self.nodes[i].consumption = nodes_input[i, 4] / 3600  # [m^3/h -> m^3/s]
            self.nodes[i].index = i
        self.x_coord_boundaries = [min(nodes_input[:, 0]), max(nodes_input[:, 0])]
        self.y_coord_boundaries = [min(nodes_input[:, 1]), max(nodes_input[:, 1])]
        self.z_coord_boundaries = [min(nodes_input[:, 2]), max(nodes_input[:, 2])]
        # Reading Tubes from File
        for i in range(len(tubes_input[:, 0])):
            self.tubes.append(Tube(tubes_input[i, 0:2]))
            self.tubes[i].length = tubes_input[i, 2]
            self.tubes[i].zeta = tubes_input[i, 3]
            self.tubes[i].index = i
            self.tubes[i].diameter = tubes_input[i, 4]
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

    def GetTubeIndexConnecting_I_with_J(self, node_i_index: int, node_j_index: int):
        for conn_tube_index in self.nodes[node_i_index].connectedTubes:
            if self.tubes[conn_tube_index].connectedNodes[0] == node_j_index \
                    or self.tubes[conn_tube_index].connectedNodes[1] == node_j_index:
                return conn_tube_index

    def Plot(self):
        plt.figure()
        plt.axes(projection='3d')
        for i in range(self.number_of_tubes):
            xs = [self.nodes[self.tubes[i].connectedNodes[0]].x, self.nodes[self.tubes[i].connectedNodes[1]].x]
            ys = [self.nodes[self.tubes[i].connectedNodes[0]].y, self.nodes[self.tubes[i].connectedNodes[1]].y]
            zs = [self.nodes[self.tubes[i].connectedNodes[0]].z, self.nodes[self.tubes[i].connectedNodes[1]].z]
            plt.plot(xs, ys, zs, 'ko-')

        plt.grid()
        plt.show()

    def DrawQ_withArrows(self):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(self.x_coord_boundaries)
        ax.set_ylim(self.y_coord_boundaries)
        ax.set_zlim(self.z_coord_boundaries)
        for i in range(self.number_of_tubes):
            n_small = min(self.tubes[i].connectedNodes)
            n_large = max(self.tubes[i].connectedNodes)

            delta_x = self.nodes[n_large].x - self.nodes[n_small].x
            delta_y = self.nodes[n_large].y - self.nodes[n_small].y
            delta_z = self.nodes[n_large].z - self.nodes[n_small].z

            if self.tubes[i].Q >= 0:
                ax.arrow3D(self.nodes[n_small].x, self.nodes[n_small].y, self.nodes[n_small].z,
                           delta_x, delta_y, delta_z,
                           mutation_scale=20,
                           arrowstyle="-|>",
                           ec='black',
                           fc='black')
            else:
                ax.arrow3D(self.nodes[n_large].x, self.nodes[n_large].y, self.nodes[n_large].z,
                           -delta_x, -delta_y, -delta_z,
                           mutation_scale=20,
                           arrowstyle="-|>",
                           ec='black',
                           fc='black')
        for node_ in self.nodes:
            if node_.consumption < 0:
                ax.scatter(node_.x, node_.y, node_.z, color='red')
            elif node_.consumption > 0:
                ax.scatter(node_.x, node_.y, node_.z, color='blue')
        plt.show()


def sign(x, y):
    if x >= y:
        return 1
    else:
        return -1


# Condition number to check if a matrix is ill-conditioned or not
# if the result is << 1 then it probably is
def matrix_state(mat: np.ndarray):
    a_ = abs(np.linalg.det(mat))
    b_ = 1
    row, col = mat.shape
    r = np.zeros(row)
    for i in range(row):
        ri_temp = 0
        for j in range(col):
            ri_temp += mat[i, j] ** 2
        r[i] = sqrt(ri_temp)
        b_ *= r[i]

    return a_ / b_
