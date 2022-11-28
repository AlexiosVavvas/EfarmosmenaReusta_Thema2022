from PipeNetworks import *

nodes, tubes = Network.Create("nodes_input.csv", "tubes_input.csv")

Network.FindConnectedTubes(nodes, tubes)
Network.FindNeighbouringElements(nodes, tubes)

Network.Plot(nodes, tubes)

# for i in range(Node.number_of_nodes):
#     print(f"Node {i} is connected with tubes: {nodes[i].connectedTubes}")
# print()
# for i in range(Node.number_of_nodes):
#     print(f"Node {i} is connected neighbours: {nodes[i].neighbouringNodes}")
