import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

def dp_link_divergence(attributes, edges):
    f = np.squeeze(np.matmul(edges, attributes) + 1e-7)
    e = np.sum(attributes, axis = 1) + 1e-7
    e = np.repeat(e, f.shape[-2], axis = 0)
    norme = np.sum(e, axis = -1, keepdims=True)
    normf = np.sum(f, axis = -1, keepdims=True)
    print(e.shape, f.shape)
    retval = entropy(e / norme, f / normf)
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

def dp_at_k(embeddings, attributes, k=10):
    k_near = k_nearest(embeddings, k=k)
    totals = attributes.sum(axis = 0) + 1e-7
    totals = totals / totals.sum(axis = -1)
    dp_total = 0
    for indices in k_near:
        distro = attributes[indices].sum(axis = 0) + 1e-7
        distro = distro / distro.sum(axis = -1)
        dp_total += entropy(totals, distro)
    return dp_total / len(embeddings)