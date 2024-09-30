import sys

parameter_lst = [0.1,0.5,0.9,1]

param_index = int(sys.argv[1])

parameter = parameter_lst[param_index]

np.save(parameter, "param.npy")


