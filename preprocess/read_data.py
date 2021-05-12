import numpy as np
from preprocess.split_data import split_train_and_test

def read_citeseer(k):
    content = open('./data/citeseer/citeseer.content')
    indexes, features, attribute_list = {}, [], []
    attr_index, attr_count = {}, 0
    for i, line in enumerate(content):
        line = line.strip().split()
        indexes[line[0]] = i
        features.append(list(map(int, line[1:-1])))
        if line[-1] not in attr_index:
            attr_index[line[-1]] = attr_count
            attr_count += 1
        attribute_list.append(attr_index[line[-1]])
    features = np.array(features)
    attributes = np.zeros((len(attribute_list, attr_count)))
    for i, item in enumerate(attribute_list):
        attributes[i, item] = 1
    
    cites = open('./data/citeseer/citeseer.cites')
    edges = np.zeros((len(features), len(features)))
    for line in cites:
        line = line.strip().split()
        edges[indexes[line[0]], indexes[line[1]]] = 1

    args = type('Args', (object,), {})
    args.fold = k
    train_edges, test_edges = split_train_and_test(args, edges)

    return features, train_edges, test_edges, attributes

def read_cora(k):
    content = open('./data/cora/cora.content')
    indexes, features, attribute_list = {}, [], []
    attr_index, attr_count = {}, 0
    for i, line in enumerate(content):
        line = line.strip().split()
        indexes[line[0]] = i
        features.append(list(map(int, line[1:-1])))
        if line[-1] not in attr_index:
            attr_index[line[-1]] = attr_count
            attr_count += 1
        attribute_list.append(attr_index[line[-1]])
    features = np.array(features)
    attributes = np.zeros((len(attribute_list, attr_count)))
    for i, item in enumerate(attribute_list):
        attributes[i, item] = 1
    
    cites = open('./data/cora/cora.cites')
    edges = np.zeros((len(features), len(features)))
    for line in cites:
        line = line.strip().split()
        edges[indexes[line[0]], indexes[line[1]]] = 1

    args = type('Args', (object,), {})
    args.fold = k
    train_edges, test_edges = split_train_and_test(args, edges)

    return features, train_edges, test_edges, attributes

def read_facebook(k):
    nodes = np.loadtxt('./data/facebook/fb_features_ego_1684.txt')
    edges = np.loadtxt('./data/facebook/fb_adjacency_1684.txt')
    features = np.concatenate([nodes[:, :147], nodes[:, 149:]], axis = -1)
    attributes = nodes[:, [147, 148]]

    args = type('Args', (object,), {})
    args.fold = k
    train_edges, test_edges = split_train_and_test(args, edges)

    return features, train_edges, test_edges, attributes

def read_pubmed(k):
    pass