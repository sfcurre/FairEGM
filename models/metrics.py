import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dp_link_divergence(attributes, edges):
    edges = sigmoid(edges)
    f = np.squeeze(np.matmul(edges, attributes) + 1e-7)
    e = np.sum(attributes, axis = 1) + 1e-7
    e = np.repeat(e, f.shape[-2], axis = 0)
    norme = np.sum(e, axis = -1, keepdims=True)
    normf = np.sum(f, axis = -1, keepdims=True)
    retval = entropy(e / norme, f / normf)
    return np.mean(retval)

def build_reconstruction_metric(pos_weight):

    def reconstruction_metric(true_adj, pred_adj):
        pred_adj = sigmoid(pred_adj)
        b_ce = -(true_adj * np.log(pred_adj + 1e-7) + (1 - true_adj) * np.log(1 - pred_adj + 1e-7))
        weight_vector = true_adj * pos_weight + (1 - true_adj)
        return np.mean(weight_vector * b_ce)

    return reconstruction_metric

def k_nearest(embeddings, k=10):
    sims = sigmoid(embeddings @ embeddings.T)
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