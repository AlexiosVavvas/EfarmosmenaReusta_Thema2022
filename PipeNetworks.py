from math import *
import matplotlib.pyplot as plt
from ArrowPlot3d import *

# Constants
ro_gas = 0.79  # density [kg/m^3]
ni_gas = 13.57e-6  # kinematic viscosity [m^2/s]
ro_air = 1.225  # density [kg/m^3]
g = 9.81  # gravity [m/s^2]
epsilon = 0.05e3  # absolute roughness [m]

i_from_name = 1.5  # το ι απο την εκφώνηση
# concurrency factors from table
f_TH_R = 0.283  # r -> type: 1 (from file)
f_ME = 0.294  # me -> type: 2 (from file)
f_TH_X = 0.8  # r -> type: 3 (from file)


class Node:

    def __init__(self, coordinates):
        self.index = None
        self.x = coordinates[0]
        self.y = coordinates[1]
        if len(coordinates) == 3:
            self.z = coordinates[2]
        else:
            self.z = 0
        self.p0 = None  # pressure [Pa]
        self.consumption = 0  # Consumption (+: inbound, -: outbound)
        self.zeta = 0
        self.connectedTubes = []
        self.neighbouringNodes = []
        self.consumer_type = None


class Tube:

    def __init__(self, connected_nodes):
        self.index = None
        self.length = None
        self.diameter = None
        self.Re = None
        self.Q = None
        self.u = None  # speed
        self.kappa = None
        self.lamda = None
        self.zeta = 0  # εντοπισμένες απώλειες στη διαδρομή
        self.zeta_from_node = 0  # εντοπισμένες απώλειες στον κόμβο εισόδου
        self.connectedNodes = list(connected_nodes.astype(int))  # index of nodes at each end


class Network:

    def __init__(self, nodes_file_dir: str, tubes_file_dir: str):
        nodes_input = np.genfromtxt(nodes_file_dir, delimiter=',', skip_header=1)
        tubes_input = np.genfromtxt(tubes_file_dir, delimiter=',', skip_header=1)
        self.nodes = []
        self.tubes = []
        self.input_consumption = 0
        # Reading Nodes from File
        for i in range(len(nodes_input[:, 0])):
            coordinates = [nodes_input[i, 0], nodes_input[i, 1], nodes_input[i, 2]]
            self.nodes.append(Node(coordinates))
            self.nodes[i].zeta = nodes_input[i, 3]
            self.nodes[i].consumer_type = nodes_input[i, 5]
            self.nodes[i].consumption = nodes_input[i, 4] / 3600  # [m^3/h -> m^3/s]
            # Differentiating C value based on consumption type
            if self.nodes[i].consumer_type == 1:
                self.nodes[i].consumption *= i_from_name * f_TH_R
            elif self.nodes[i].consumer_type == 3:
                self.nodes[i].consumption *= i_from_name * f_TH_X
            elif self.nodes[i].consumer_type == 2:
                self.nodes[i].consumption *= i_from_name * f_ME
            self.input_consumption += self.nodes[i].consumption
            self.nodes[i].index = i
        self.input_consumption = abs(self.input_consumption)
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
        self.max_p0_drop = None
        self.max_p_drop = None

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

    # noinspection PyPep8Naming
    def SolveNR(self, ITER, relax, deltaP_stop, plot_flag: bool, print_flag: bool):
        if print_flag:
            print()
            print(f"------------- NEWTON - RAPHSON -------------")

        # Manually initialize total pressures
        p_entrance = 0.025e5  # [Pa] (static pressure)
        self.nodes[0].p0 = p_entrance + 8 * ro_gas * (self.input_consumption ** 2) / \
                           ((pi ** 2) * (self.tubes[self.nodes[0].connectedTubes[0]].diameter ** 4)) + \
                           (ro_gas - ro_air) * g * self.nodes[0].z
        p_drop_step = 0.2 * p_entrance / self.number_of_nodes
        for i in range(1, self.number_of_nodes):
            self.nodes[i].p0 = abs(p_entrance - p_drop_step * i)

        # Calculating Tube Kappa
        for tube_ in self.tubes:
            tube_.lamda = (1 / (1.14 - 2 * log10(epsilon / tube_.diameter))) ** 2  # Jain's Equation for Re->Inf
            tube_.kappa = (tube_.lamda * tube_.length / tube_.diameter + tube_.zeta + tube_.zeta_from_node) * \
                          (8 * ro_gas) / ((pi ** 2) * (tube_.diameter ** 4))

        # Newton-Raphson Method
        F = np.zeros(self.number_of_nodes)
        DF = np.zeros((self.number_of_nodes, self.number_of_nodes))  # Jacobian Matrix
        DF[0, 0] = 1
        deltaP = np.zeros(self.number_of_nodes)
        deltaP_old = np.zeros(self.number_of_nodes)

        p0_guess = np.zeros(self.number_of_nodes)
        for i, node in enumerate(self.nodes):
            p0_guess[i] = node.p0

        iter_count = 0
        first_round = True
        while (((sqrt(sum(abs(deltaP)))) / self.number_of_nodes > deltaP_stop) or first_round) & (
                iter_count < ITER):

            # Finding F() values
            for i in range(1, self.number_of_nodes):
                temp_F_sum = 0
                for j in self.nodes[i].neighbouringNodes:
                    kappa = self.tubes[self.GetTubeIndexConnecting_I_with_J(i, j)].kappa
                    temp_F_sum += sign(p0_guess[j], p0_guess[i]) * sqrt(abs(p0_guess[i] - p0_guess[j]) / kappa)
                F[i] = temp_F_sum + self.nodes[i].consumption

            # Calculating derivatives DF(i, j)
            for i in range(1, self.number_of_nodes):
                temp_DF_sum = 0
                for j in self.nodes[i].neighbouringNodes:
                    kappa = self.tubes[self.GetTubeIndexConnecting_I_with_J(i, j)].kappa
                    temp_DF = 1 / 2 / sqrt(kappa) / sqrt(abs(p0_guess[i] - p0_guess[j]))
                    DF[i, j] = temp_DF
                    temp_DF_sum += temp_DF

                DF[i, i] = -temp_DF_sum

            # Solving system
            deltaP = np.linalg.solve(DF, -F)
            deltaP = (1 - relax) * deltaP_old + deltaP * relax
            p0_guess_old = p0_guess
            p0_guess = p0_guess + deltaP

            # plotting
            if plot_flag:
                for node2plot in range(0, self.number_of_nodes):
                    plt.plot([iter_count, iter_count + 1], [p0_guess_old[node2plot], p0_guess[node2plot]], 'k-',
                             linewidth=0.2)

            iter_count += 1
            first_round = False

        if print_flag:
            print(f"Loop Finished in {iter_count}/{ITER} iterations "
                  f"while error handling returns <<{((sqrt(sum(abs(deltaP)))) / self.number_of_nodes > deltaP_stop)}>>\n")

        # Saving converged results to network object variables
        for i, node in enumerate(self.nodes):
            node.p0 = p0_guess[i]
            if print_flag:
                print(f"Node {i} has P0 = {node.p0:.5f}\tand final dP0 = {deltaP[i]}")
        if print_flag: print()

        # Calculating Q (+: from lower node index to higher)
        #   e.g. if Q connecting node 2 and 3 is <0
        #        that means that if flows from 3 -> 2
        for tube_ in self.tubes:
            i = tube_.connectedNodes[0]
            j = tube_.connectedNodes[1]
            Q = sign(self.nodes[i].p0, self.nodes[j].p0) * \
                sqrt(abs(self.nodes[i].p0 - self.nodes[j].p0) / tube_.kappa)
            if i > j:
                Q *= -1
            tube_.Q = Q
            tube_.u = 8 * Q ** 2 * ro_gas / (pi ** 2 * tube_.diameter ** 4)

            if print_flag:
                print(f"Tube {tube_.index} (nodes {i} - {j}) diameter : {tube_.diameter * 1000:.2f} [mm]\thas\t"
                      f"Q = {Q * 3600:.4f} [m^3/h]\t-> u = {tube_.u:.4f} [m/s]")

        # Finding static pressures
        p_static = []
        for tube_ in self.tubes:
            for i in range(2):
                node_ind = tube_.connectedNodes[i]
                p_static.append(self.nodes[node_ind].p0 -
                                8 * ro_gas * (tube_.Q ** 2) / ((pi ** 2) * (tube_.diameter ** 4)) -
                                (ro_gas - ro_air) * g * self.nodes[node_ind].z)

        # showing final plot
        if plot_flag:
            plt.title("Newton-Raphson Method")
            plt.xlabel("Iterations")
            plt.ylabel("Total Pressures [Pa]")
            plt.grid()
            plt.draw()
            plt.show()

        # max pressure drop calc
        self.max_p0_drop = (max(p0_guess) - min(p0_guess))  # [Pa]
        self.max_p_drop = (max(p_static) - min(p_static))  # [Pa]
        if print_flag:
            print(f"Max total pressure drop: {self.max_p0_drop / 100:.8f} [mBar]")
            print(f"Max static pressure drop: {self.max_p_drop / 100:.8f} [mBar]")
            print()

        # Return whether method converged or not
        if iter_count < ITER:
            return False
        else:
            return True


def sign(x, y):
    if x >= y:
        return 1
    else:
        return -1
