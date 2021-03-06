import numpy as np
import TVGL

data = np.loadtxt("testdata.txt")
alpha = 5
beta = 10
penalty = "L1"
slice_size = 20
Theta, S = TVGL.solve(data, alpha, beta, penalty, slice_size)
