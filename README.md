# CUDACuts
Based on CUDA Cuts Code
NPPI library remove graphcut algorithm after CUDA version 7.0.
This project aims to give a fast version of graphcut algorithm on CUDA.

### TODO
1. optimaze cuda cuts algorihtms in the paper
2. support complicated graph structure (not only plain images)

### data format

The first line contains three numbers describe the data size:
width ***W*** height ***H***, number of labels ***L***

flowing ***H*** lines contains ***W*** integers which describes the ground truth data.
(Not used in this code)

flowing ***H L*** lines contains ***W*** integers between label vertex. Here ***L = 2***, the data describes the values of edges between
pixels and source vertex, pixels and target vertex.

flowing ***H*** lines contains ***W - 1*** integers which describes the values of horizontal edges.

flowing ***H - 1*** lines contains ***W*** integers which describes the values of vertical edges.
