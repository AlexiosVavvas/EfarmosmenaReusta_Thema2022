from math import *
from PipeNetworks import *

# Constants
ro_gas = 0.79  # [kg/m^3]
ni_gas = 13.57e-6  # [m^2/s]
ro_air = 1.225  # [kg/m^3]
g = 9.81  # gravity [m/s^2]
epsilon = 0.05e3  # absolute roughness [m]

# Creating network from file
my_network = Network("nodes_input.csv", "tubes_input.csv")

# Analyzing Topology
my_network.FindConnectedTubes()
my_network.FindNeighbouringElements()

# Manually initialize static pressures
my_network.nodes[0].p_static = 0.021e5
p_drop_step = (0.021e5 - 0.019e5) / my_network.number_of_nodes
for i in range(1, my_network.number_of_nodes):
    my_network.nodes[i].p_static = abs(0.021e5 - p_drop_step * (my_network.nodes[i].z + i))

# Initial Diameter Guess
d_0 = 0.05  # [m]
for tube_ in my_network.tubes:
    tube_.lamda = (1 / (1.14 - 2 * log10(epsilon / d_0))) ** 2
    tube_.diameter = d_0
    tube_.kappa = tube_.lamda * tube_.length / tube_.diameter + tube_.zeta

# Newton-Raphson Method
F = np.zeros(my_network.number_of_nodes)
DF = np.zeros((my_network.number_of_nodes, my_network.number_of_nodes))  # Jacobian Matrix
deltaP = np.ones(my_network.number_of_nodes)
deltaP_old = np.ones(my_network.number_of_nodes)

DF[0, 0] = 1

p0_guess = np.zeros(my_network.number_of_nodes)
for i, node in enumerate(my_network.nodes):
    p0_guess[i] = node.p_static + ro_gas * g * node.z

ITER = 400
relax = 0.01

iter_count = 0
while (abs(deltaP) > 0.01).any() & (iter_count < ITER):

    # Finding F() values
    for i in range(1, my_network.number_of_nodes):
        temp_F_sum = 0
        for j in my_network.nodes[i].neighbouringNodes:
            kappa = my_network.tubes[my_network.GetTubeIndexConnectingIwithJ(i, j)].kappa
            temp_F = sign(p0_guess[j], p0_guess[i]) * sqrt(
                abs(p0_guess[j] - p0_guess[i]) / kappa)
            temp_F_sum += temp_F
        F[i] = temp_F_sum + my_network.nodes[i].consumption

    # Calculating derivatives DF(i, j)
    for i in range(1, my_network.number_of_nodes):
        temp_DF = 0
        temp_DF_sum = 0
        for j in my_network.nodes[i].neighbouringNodes:
            if i != j:
                kappa = my_network.tubes[my_network.GetTubeIndexConnectingIwithJ(i, j)].kappa
                temp_DF = 1 / 2 / sqrt(kappa) / sqrt(abs(p0_guess[j] - p0_guess[i]))
                DF[i, j] = temp_DF
                temp_DF_sum += temp_DF

        DF[i, i] = -temp_DF_sum

    print(f"matrix DF at iter {iter_count} has det = {np.linalg.det(DF)}")
    # Solving system
    deltaP = np.linalg.solve(DF, -F)
    deltaP = (1 - relax) * deltaP_old + deltaP * relax
    deltaP[0] = 0
    p0_guess = p0_guess + deltaP

    iter_count += 1

print(f"Loop Finished in {iter_count} iterations while error handling returns <<{(abs(deltaP) > 0.001).any()}>>\n")

for i, node in enumerate(my_network.nodes):
    node.p0 = p0_guess[i]
    print(f"Node {i} has P0 = {node.p0:.4f} and final dP = {deltaP[i]}")
