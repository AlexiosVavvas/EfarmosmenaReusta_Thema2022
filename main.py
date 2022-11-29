from math import *
from PipeNetworks import *

# Creating network from file
my_network = Network("nodes_input.csv", "tubes_input.csv")

# Analyzing Topology
my_network.FindConnectedTubes()
my_network.FindNeighbouringElements()

# Manually initialize pressures
my_network.nodes[0].p0 = 0.021e5
init_p_drop_step = (0.021e5 - 0.001e5) / my_network.number_of_nodes
for i in range(1, my_network.number_of_nodes):
    my_network.nodes[i].p0 = abs(0.021e5 - init_p_drop_step * i)

# Initial Diameter Guess
d_0 = 0.04  # [m]
epsilon = 0.05e3  # [m]
for tube_ in my_network.tubes:
    tube_.lamda = (1 / (1.14 - 2 * log10(epsilon / d_0))) ** 2
    tube_.diameter = d_0
    tube_.kappa = tube_.lamda * tube_.length / tube_.diameter + tube_.zeta

# Newton-Raphson Method
F = np.zeros(my_network.number_of_nodes)
DF = np.zeros((my_network.number_of_nodes, my_network.number_of_nodes))
deltaP = np.ones(my_network.number_of_nodes)

P_guess = np.zeros(my_network.number_of_nodes)
P_guess_old = np.copy(P_guess)
# error = np.ones(my_network.number_of_nodes)

for i, node in enumerate(my_network.nodes):
    P_guess[i] = node.p0

ITER = 100
relax = 0.01

iter_count = 0
while (abs(deltaP) > 0.001).any() & (iter_count < ITER):

    # Finding F() values
    for i in range(my_network.number_of_nodes):
        temp_F_sum = 0
        for j in my_network.nodes[i].neighbouringNodes:
            kappa = my_network.tubes[my_network.GetTubeIndexConnectingIwithJ(i, j)].kappa
            temp_F = sign(P_guess[j], P_guess[i]) * sqrt(
                abs(P_guess[j] - P_guess[i]) / kappa)
            temp_F_sum += temp_F
        F[i] = temp_F_sum

    # Calculating derivatives DF(i, j)
    for i in range(my_network.number_of_nodes):
        temp_DF = 0
        temp_DF_sum = 0
        for j in my_network.nodes[i].neighbouringNodes:
            if i != j:
                kappa = my_network.tubes[my_network.GetTubeIndexConnectingIwithJ(i, j)].kappa
                temp_DF = 1 / 2 / sqrt(kappa) / sqrt(abs(P_guess[j] - P_guess[i]))
                DF[i, j] = temp_DF
                temp_DF_sum += temp_DF

        DF[i, i] = -temp_DF_sum

    print(f"matrix DF at iter {iter_count} has det = {np.linalg.det(DF)}")
    deltaP = np.linalg.solve(DF, F)
    P_guess = (1 - relax) * P_guess_old + deltaP * relax
    # error = P_guess - P_guess_old
    P_guess_old = P_guess

    # print(P_guess)
    print(f"Done one loop, i: {iter_count} -> {iter_count + 1}")
    iter_count += 1

print(f"Loop Finished in {iter_count} iterations while error handling returns <<{(abs(deltaP) > 0.001).any()}>>\n")

for i, node in enumerate(my_network.nodes):
    node.p0 = P_guess[i]
    print(f"Node {i} has P0 = {node.p0:.4f}")
print(deltaP)
