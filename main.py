from PipeNetworks import *

# Creating network from file
my_network = Network("nodes_input.csv", "tubes_input.csv")

# Analyzing Topology
my_network.FindConnectedTubes()
my_network.FindNeighbouringElements()

# First solve network with NR
my_network.SolveNR(300, 0.1, 10 ** -5, False, False)

# Getting standard diameters from file
std_diam = np.genfromtxt("standard_diameters.csv", delimiter=',', skip_header=1)
std_diam_len = len(std_diam)

# Checking and Changing
speed_goal = 4  # [m/s]
max_p0_drop_goal = 200  # [Pa]

for i0 in range(50):

    for tube_ in my_network.tubes:
        old_tube_ind = np.where((tube_.diameter - 0.003 < std_diam) * (std_diam < tube_.diameter + 0.003))[0][0]
        if tube_.u > speed_goal:
            new_tube_ind = old_tube_ind + 1
            if new_tube_ind >= std_diam_len:
                tube_.diameter = std_diam[std_diam_len - 1]
            elif new_tube_ind <= 0:
                tube_.diameter = std_diam[0]
            else:
                tube_.diameter = std_diam[new_tube_ind]
        if tube_.u < speed_goal:
            new_tube_ind = old_tube_ind - 1
            if new_tube_ind >= std_diam_len:
                tube_.diameter = std_diam[std_diam_len - 1]
            elif new_tube_ind <= 0:
                tube_.diameter = std_diam[0]
            else:
                tube_.diameter = std_diam[new_tube_ind]

    my_network.SolveNR(300, 0.1, 10**-5, False, True)

