from PipeNetworks import *

# Creating network from file
my_network = Network("nodes_input.csv", "tubes_input.csv")

# Analyzing Topology
my_network.FindConnectedTubes()
my_network.FindNeighbouringElements()

# First solve network with NR
my_network.SolveNR(300, 0.1, 10 ** -5, False, True)
# my_network.DrawQ_withArrows()
# p_drop_0 = my_network.max_p0_drop

# Getting standard diameters from file
std_diam = np.genfromtxt("standard_diameters.csv", delimiter=',', skip_header=1)
std_diam_len = len(std_diam)

# Checking and Changing
speed_goal = 4  # [m/s]
# max_p0_drop_goal = 200  # [Pa]

for i0 in range(8):

    for tube_ in my_network.tubes:

        # Change diameter according to speed, to match requirement
        old_tube_ind = np.where((tube_.diameter - 0.003 < std_diam) * (std_diam < tube_.diameter + 0.003))[0][0]
        if tube_.u > speed_goal * 1.3:
            new_tube_ind = old_tube_ind + 1
            if new_tube_ind >= std_diam_len - 1:
                tube_.diameter = std_diam[std_diam_len - 1]
            elif new_tube_ind <= 0:
                tube_.diameter = std_diam[0]
            else:
                tube_.diameter = std_diam[new_tube_ind]
        elif tube_.u < speed_goal * 0.7:
            new_tube_ind = old_tube_ind - 1
            if new_tube_ind >= std_diam_len - 1:
                tube_.diameter = std_diam[std_diam_len - 1]
            elif new_tube_ind <= 0:
                tube_.diameter = std_diam[0]
            else:
                tube_.diameter = std_diam[new_tube_ind]

        # Adding friction from beginning node
        if tube_.Q >= 0:
            tube_.zeta_from_node = my_network.nodes[min(tube_.connectedNodes)].zeta
        else:
            tube_.zeta_from_node = my_network.nodes[max(tube_.connectedNodes)].zeta

    # Solve with NR
    my_network.SolveNR(300, 0.1, 10 ** -5, False, True)

# print((my_network.max_p0_drop - p_drop_0)/p_drop_0*100)
