import numpy as np
from scipy.stats import entropy

def dp_link_divergence(attributes, edges):
    #edges is (batch, filters, nodes, nodes)
    #attributes is (batch, filters, nodes, attributes)
    f = np.matmul(edges, attributes)
    #f is (batch, filters, nodes, attributes)
    #f is count of edges per sensitive class
    f = f / np.sum(attributes, axis = 2)[..., None, :]
    #f is proportion per node
    e = np.ones_like(f)
    e = np.clip(e, 1e-7, 1)
    f = np.clip(f, 1e-7, 1)
    retval = entropy(e / e.sum(axis = -1)[...,None],
                     f / f.sum(axis = -1)[...,None], axis = -1)
    return retval