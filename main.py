from PipeNetworks import *

# Constants
ro_gas = 0.79  # density [kg/m^3]
ni_gas = 13.57e-6  # kinematic viscosity [m^2/s]
ro_air = 1.225  # density [kg/m^3]
g = 9.81  # gravity [m/s^2]
epsilon = 0.05e3  # absolute roughness [m]

# Creating network from file
my_network = Network("nodes_input.csv", "tubes_input.csv")

# Analyzing Topology
my_network.FindConnectedTubes()
my_network.FindNeighbouringElements()

# Manually initialize static pressures
p_entrance = 0.025e5
my_network.nodes[0].p_static = p_entrance
p_drop_step = 0.2 * p_entrance / my_network.number_of_nodes
for i in range(1, my_network.number_of_nodes):
    my_network.nodes[i].p_static = abs(p_entrance - p_drop_step * i)

# Calculating Tube Kappa
for tube_ in my_network.tubes:
    tube_.lamda = (1 / (1.14 - 2 * log10(epsilon / tube_.diameter))) ** 2  # Jain's Equation for Re->Inf
    tube_.kappa = tube_.lamda * tube_.length / tube_.diameter

# Newton-Raphson Method
F = np.zeros(my_network.number_of_nodes)
DF = np.zeros((my_network.number_of_nodes, my_network.number_of_nodes))  # Jacobian Matrix
DF[0, 0] = 1
deltaP = np.zeros(my_network.number_of_nodes)
deltaP_old = np.zeros(my_network.number_of_nodes)

p0_guess = np.zeros(my_network.number_of_nodes)
for i, node in enumerate(my_network.nodes):
    p0_guess[i] = node.p_static

ITER = 200
relax = 0.1
plot_flag = True
deltaP_stop = 10 ** -4


iter_count = 0
first_round = True
while (((sqrt(sum(abs(deltaP)))) / my_network.number_of_nodes > deltaP_stop) or first_round) & (iter_count < ITER):

    # Finding F() values
    for i in range(1, my_network.number_of_nodes):
        temp_F_sum = 0
        for j in my_network.nodes[i].neighbouringNodes:
            kappa = my_network.tubes[my_network.GetTubeIndexConnecting_I_with_J(i, j)].kappa
            temp_F_sum += sign(p0_guess[j], p0_guess[i]) * sqrt(abs(p0_guess[i] - p0_guess[j]) / kappa)
        F[i] = temp_F_sum + my_network.nodes[i].consumption

    # Calculating derivatives DF(i, j)
    for i in range(1, my_network.number_of_nodes):
        temp_DF = 0
        temp_DF_sum = 0
        for j in my_network.nodes[i].neighbouringNodes:
            kappa = my_network.tubes[my_network.GetTubeIndexConnecting_I_with_J(i, j)].kappa
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
        for node2plot in range(0, my_network.number_of_nodes):
            plt.plot([iter_count, iter_count + 1], [p0_guess_old[node2plot], p0_guess[node2plot]], 'k-', linewidth=0.2)

    iter_count += 1
    first_round = False
print(f"Loop Finished in {iter_count}/{ITER} iterations "
      f"while error handling returns <<{((sqrt(sum(abs(deltaP)))) / my_network.number_of_nodes > deltaP_stop)}>>\n")

# Saving converged results to network object variables
p_static = np.zeros(my_network.number_of_nodes)
for i, node in enumerate(my_network.nodes):
    node.p0 = p0_guess[i]
    node.p_static = node.p0 - (ro_gas - ro_air) * g * node.z
    p_static[i] = node.p_static
    print(f"Node {i} has P0 = {node.p0:.5f}, \tP_static = {node.p_static} \tand final dP0 = {deltaP[i]}")

# showing final plot
if plot_flag:
    plt.title("Newton-Raphson Method")
    plt.xlabel("Iterations")
    plt.ylabel("Total Pressures [Pa]")
    plt.grid()
    plt.draw()
    plt.show()

# max pressure drop calc
print(f"Max total pressure drop: {(max(p0_guess) - min(p0_guess)) / 100:.8f} [mBar]")
print(f"Max static pressure drop: {(max(p_static) - min(p_static)) / 100:.8f} [mBar]")
print()

# Calculating Q (+: from lower node index to higher)
#   e.g. if Q connecting node 2 and 3 is <0
#        that means that if flows from 3 -> 2
for tube_ in my_network.tubes:
    i = tube_.connectedNodes[0]
    j = tube_.connectedNodes[1]
    Q = sign(my_network.nodes[i].p0, my_network.nodes[j].p0) * \
        sqrt(abs(my_network.nodes[i].p0 - my_network.nodes[j].p0) / tube_.kappa)
    if i > j:
        Q *= -1
    tube_.Q = Q

    print(f"Tube {tube_.index} (nodes {i} - {j}) has "
          f"\tQ = {Q * 3600:.4f} [m^3/h]\t-> u = {8 * Q ** 2 * ro_gas / (pi ** 2 * tube_.diameter ** 4):.4f} [m/s]")

my_network.DrawQ_withArrows()
