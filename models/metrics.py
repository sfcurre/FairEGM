import numpy as np
from scipy.stats import entropy
from collections import defaultdict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dp_link_divergence(attributes, edges):
    edges = sigmoid(edges)
    f = np.squeeze(np.matmul(edges, attributes) + 1e-7)
    e = np.sum(attributes, axis = 1) + 1e-7
    e = np.repeat(e, f.shape[-2], axis = 0)
    norme = np.sum(e, axis = -1, keepdims=True)
    normf = np.sum(f, axis = -1, keepdims=True)
    retval = entropy(e / norme, f / normf, axis = -1)
    return retval.mean()

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

def dyf_at_threshold(attributes, prob_list, global_attributes, percent=10):
    percentile = np.percentile(prob_list, 100-percent)
    chosen = (prob_list >= percentile)
    attribute_distro = global_attributes.sum(axis=0)
    upr_indices = np.triu_indices(len(attribute_distro))
    global_count = (attribute_distro[...,None] @ attribute_distro[None,...])[upr_indices] + 1 #expand to calc pairs
    chosen_count = np.zeros((len(attribute_distro), len(attribute_distro)))
    for (a, b) in attributes[chosen]:
        if a <= b:
            chosen_count[a, b] += 1
        else:
            chosen_count[b, a] += 1
    chosen_count = chosen_count[upr_indices] + 1
    global_distro = global_count / global_count.sum()
    chosen_distro = chosen_count / chosen_count.sum()
    return entropy(global_distro, chosen_distro)


def dp_threshold(embeddings, attributes, percent=10, train_adj=None, weighted=False):
    adj = sigmoid(embeddings @ embeddings.T)
    if train_adj is not None:
        adj = adj - train_adj
    percentile = np.percentile(adj, 100-percent)
    chosen = (adj >= percentile) - np.eye(len(embeddings))
    totals = attributes.sum(axis = 0) + 1e-7
    totals = totals / totals.sum(axis = -1)
    dp_total = 0
    total_weight = 0
    for indices in chosen:
        weight = 1 if not weighted else indices.sum()
        distro = attributes[indices.astype(bool)].sum(axis = 0) + 1e-7
        distro = distro / distro.sum(axis = -1)
        dp_total += entropy(totals, distro) * weight
        total_weight += weight
    return dp_total / total_weight

def delta_dp_threshold(attributes, prob_list, percent=10):
    percentile = np.percentile(prob_list, 100-percent)
    chosen = (prob_list >= percentile)
    return dp(attributes[chosen], prob_list[chosen])

def max_p_diff(attributes, prob_list):
    pdict = defaultdict(float)
    cdict = defaultdict(int)
    for p, (a, b) in zip(prob_list, attributes):
            cdict[a, b] += 1
            cdict[b, a] += 1
            pdict[a, b] += p
            pdict[b, a] += p
    mx = 0
    mn = 1
    for key in cdict:
            p = pdict[key] / cdict[key]
            if p < mn:
                mn = p
            if p > mx:
                mx = p
    return mx - mn

def dp(attributes, prob_list):
    intra_probs, inter_probs = [], []
    for p, (a, b) in zip(prob_list, attributes):
        if a == b:
            intra_probs.append(p)
        else:
            inter_probs.append(p)
    return abs(sum(intra_probs) / len(intra_probs) - sum(inter_probs) / len(inter_probs))