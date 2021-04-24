import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

def dp_link_divergence(attributes, edges):
    #edges is (batch, nodes, nodes)
    #attributes is (batch, nodes, attributes)
    f = np.matmul(edges, attributes)
    #f is (batch, nodes, attributes)
    #f is count of edges per sensitive class
    f = f / np.sum(attributes, axis = 1)[..., None, :]
    #f is proportion per node
    e = np.ones_like(f)
    f = np.clip(f, 1e-7, 1)
    retval = entropy(e / e.sum(axis = -1)[...,None],
                     f / f.sum(axis = -1)[...,None], axis = -1)
    return retval

def k_nearest(embeddings, k=10):
    sims = 1 - squareform(pdist(embeddings, metric='cosine'))
    order = np.argsort(sims, axis =-1)[:, :-1] #exclude self
    return order[:, -k:]

def recall_at_k(embeddings, test_edges, k=10):
    k_near = k_nearest(embeddings, k=k)
    edges = np.argwhere(test_edges == 1)
    rec_count = 0
    for i, j in edges:
        if j in k_near[i]:
            rec_count += 1
    return rec_count / len(edges)

def dp_at_k_dif(embeddings, attributes, k=10):
    k_near = k_nearest(embeddings, k=k)
    totals = attributes.sum(axis = 0)
    dp_total = 0
    for indices in k_near:
        distro = attributes[indices].sum(axis = 0) / totals
        distro = np.clip(distro, 1e-7, 1)
        dp_total += abs(distro[1] - distro[0])
    return dp_total / len(embeddings)

def dp_at_k_div(embeddings, attributes, k=10):
    k_near = k_nearest(embeddings, k=k)
    totals = attributes.sum(axis = 0)
    unif = np.ones(attributes.shape[-1]) / attributes.shape[-1]
    dp_total = 0
    for indices in k_near:
        distro = attributes[indices].sum(axis = 0) / totals
        distro = np.clip(distro, 1e-7, 1)
        distro = distro / distro.sum()
        dp_total += entropy(unif, distro)
    return dp_total / len(embeddings)