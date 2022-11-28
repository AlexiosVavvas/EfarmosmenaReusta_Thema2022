from PipeNetworks import *

my_network = Network("nodes_input.csv", "tubes_input.csv")

my_network.FindConnectedTubes()
my_network.FindNeighbouringElements()

my_network.Plot()
# for i in range(Node.number_of_nodes):
#     print(f"Node {i} is connected with tubes: {nodes[i].connectedTubes}")
# print()
# for i in range(Node.number_of_nodes):
#     print(f"Node {i} is connected neighbours: {nodes[i].neighbouringNodes}")
