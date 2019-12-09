# Time-varying Graphical Lasso
Time-varying Graphical Lasso(TVGL) is a python solver for time-varying network inferring.

# Download
```
git clone https://github.com/ams129/TVGL.git
```

# Usage
TVGL can be called through the following file:
```
TVGL.py
```

Parameters
* data : numpy array with the raw data
* block_size : Number of samples in each timestamps
* alpha : the regularization parameter controlling the network sparsity
* beta : the beta parameter controlling the temporal consistency

# Example
Running the following script provides an example of how the TVGL solver can be used:
```
sample.py
```
