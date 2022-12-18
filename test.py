import numpy as np

# Getting consumption variance from a file
cons_var = np.genfromtxt("consumption_variance.csv", delimiter=',', skip_header=1)
cons_var_len = len(cons_var)

for i in range(cons_var_len):
    print(cons_var[i])