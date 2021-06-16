import numpy as np
import tvgl

# load data
X = np.loadtxt("testdata.txt")

# set parameters
alpha = 10
beta = 10
penalty_type = "L1"
slice_size = 20

# run TVGL
model = tvgl.TVGL(alpha, beta, penalty_type, slice_size)
model.fit(X)
