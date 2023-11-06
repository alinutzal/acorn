import cupy as cp
from pylibraft.common import DeviceResources
from pylibraft.neighbors.brute_force import knn
n_samples = 50000
n_features = 50
n_queries = 1000
dataset = cp.random.random_sample((n_samples, n_features),
                                  dtype=cp.float32)
# Search using the built index
queries = cp.random.random_sample((n_queries, n_features),
                                  dtype=cp.float32)
k = 40
distances, neighbors = knn(dataset, queries, k)
distances = cp.asarray(distances)
neighbors = cp.asarray(neighbors)
print(distances)
print(neighbors)

import cupy as cp
from pylibraft.common import DeviceResources
from pylibraft.neighbors import cagra
n_samples = 50000
n_features = 50
n_queries = 1000
k = 10
dataset = cp.random.random_sample((n_samples, n_features),
                                  dtype=cp.float32)
handle = DeviceResources()
build_params = cagra.IndexParams(metric="sqeuclidean")
index = cagra.build(build_params, dataset, handle=handle)
distances, neighbors = cagra.search(cagra.SearchParams(),
                                     index, dataset,
                                     k, handle=handle)
# pylibraft functions are often asynchronous so the
# handle needs to be explicitly synchronized
handle.sync()
distances = cp.asarray(distances)
neighbors = cp.asarray(neighbors)
print(distances)
print(neighbors)

