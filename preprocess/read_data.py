import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from preprocess.split_data import *

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values)/(max_values-min_values) - 1

def read_bail(k, r=0):
    features = pd.read_csv('./data/bail/bail.csv')
    edge_list = open('./data/bail/bail_edges.txt')

    attribute_series = features.pop('WHITE')
    attributes = np.zeros((len(attribute_series), attribute_series.max() + 1))
    for i, attr in enumerate(attribute_series):
        attributes[i][attr] = 1

    def convert_sci(note):
        base, power = note.split('e+')
        return int(float(base) * (10 ** int(power)))

    edges = np.zeros((len(features), len(features)))
    for line in edge_list:
        line = line.strip().split()
        edges[convert_sci(line[0]), convert_sci(line[1])] = 1
        edges[convert_sci(line[1]), convert_sci(line[0])] = 1

    if k == 0:
        return feature_norm(features.values), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)
        
    return feature_norm(features.values), edge_gen, attributes

def read_citeseer(k, r=0):
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
    attributes = np.zeros((len(attribute_list), attr_count))
    for i, item in enumerate(attribute_list):
        attributes[i, item] = 1
    
    cites = open('./data/citeseer/citeseer.cites')
    edges = np.zeros((len(features), len(features)))
    for line in cites:
        line = line.strip().split()
        try:
            edges[indexes[line[0]], indexes[line[1]]] = 1
        except KeyError:
            continue

    if k == 0:
        return feature_norm(features), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)
        
    return feature_norm(features), edge_gen, attributes

def read_cora(k, r=0):
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
    attributes = np.zeros((len(attribute_list), attr_count))
    for i, item in enumerate(attribute_list):
        attributes[i, item] = 1
    
    cites = open('./data/cora/cora.cites')
    edges = np.zeros((len(features), len(features)))
    for line in cites:
        line = line.strip().split()
        edges[indexes[line[0]], indexes[line[1]]] = 1

    if k == 0:
        return feature_norm(features), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)
        
    return feature_norm(features), edge_gen, attributes

def read_credit(k, r=0):
    features = pd.read_csv('./data/credit/credit.csv')
    edge_list = open('./data/credit/credit_edges.txt')

    features.pop('Single')
    attribute_series = features.pop('Age')
    attributes = np.zeros((len(attribute_series), attribute_series.max() + 1))
    for i, attr in enumerate(attribute_series):
        attributes[i][attr] = 1

    def convert_sci(note):
        base, power = note.split('e+')
        return int(float(base) * (10 ** int(power)))

    edges = np.zeros((len(features), len(features)))
    for line in edge_list:
        line = line.strip().split()
        edges[convert_sci(line[0]), convert_sci(line[1])] = 1
        edges[convert_sci(line[1]), convert_sci(line[0])] = 1

    if k == 0:
        return feature_norm(features.values), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)
        
    return feature_norm(features.values), edge_gen, attributes

def read_facebook(k, r=0):
    nodes = np.loadtxt('./data/facebook/fb_features_ego_1684.txt')
    edges = np.loadtxt('./data/facebook/fb_adjacency_1684.txt')
    features = np.concatenate([nodes[:, :147], nodes[:, 149:]], axis = -1)
    attributes = nodes[:, [147, 148]]

    if k == 0:
        return feature_norm(features), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)
        
    return feature_norm(features), edge_gen, attributes

def read_german(k, r=0):
    features = pd.read_csv('./data/german/german.csv')
    edge_list = open('./data/german/german_edges.txt')

    features.pop('OtherLoansAtStore')
    features.pop('PurposeOfLoan')
    attribute_series = features.pop('Gender')
    attribute_series[attribute_series == 'Female'] = 1
    attribute_series[attribute_series == 'Male'] = 0
    attributes = np.zeros((len(attribute_series), attribute_series.max() + 1))
    for i, attr in enumerate(attribute_series):
        attributes[i][attr] = 1

    def convert_sci(note):
        base, power = note.split('e+')
        return int(float(base) * (10 ** int(power)))

    edges = np.zeros((len(features), len(features)))
    for line in edge_list:
        line = line.strip().split()
        edges[convert_sci(line[0]), convert_sci(line[1])] = 1
        edges[convert_sci(line[1]), convert_sci(line[0])] = 1

    if k == 0:
        return feature_norm(features.values), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)

    return feature_norm(features.values), edge_gen, attributes

def read_pubmed(k, r=0):
    content = open('./data/pubmed/Pubmed-Diabetes.NODE.paper.tab')
    #skip two header lines
    content.readline()
    content.readline()
    indexes, attributes = {}, []
    feature_list, feat_index, feat_count = [], {}, 0
    for i, line in enumerate(content):
        line = line.strip().split()
        indexes[line[0]] = i
        
        attr = np.zeros(3)
        attr[int(line[1].split('=')[1]) - 1] = 1
        attributes.append(attr)
        
        feats = {}
        for f in line[2:-1]:
            feat_label, val = f.split('=')
            if feat_label not in feat_index:
                feat_index[feat_label] = feat_count
                feat_count += 1
            feats[feat_label] = val
        feature_list.append(feats)

    features, attributes = np.zeros((len(feature_list), feat_count)), np.array(attributes)
    for i, feats in enumerate(feature_list):
        for key, val in feats.items():
            features[i][feat_index[key]] = val

    edge_list = open('./data/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab')
    #skip two header lines
    edge_list.readline()
    edge_list.readline() 
    edges = np.zeros((len(features), len(features)))
    for line in edge_list:
        line = line.strip().split()
        p1, p2 = line[1].split(':')[1], line[3].split(':')[1]
        edges[indexes[p1], indexes[p2]] = 1

    if k == 0:
        return feature_norm(features), edges, attributes

    args = type('Args', (object,), {})
    args.fold = k
    args.random = r
    if args.random:
        edge_gen = split_train_and_test_random(args, edges)
    else:
        edge_gen = split_train_and_test(args, edges)

    return feature_norm(features), edge_gen, attributes

def read_data(dataset, folds, r=0):
    if dataset == 'bail':
        get_data = read_bail(folds, r=r)
    elif dataset == 'citeseer':
        get_data = read_citeseer(folds, r=r)
    elif dataset == 'cora':
        get_data = read_cora(folds, r=r)
    elif dataset == 'credit':
        get_data = read_credit(folds, r=r)
    elif dataset == 'facebook':
        get_data = read_facebook(folds, r=r)
    elif dataset == 'german':
        get_data = read_german(folds, r=r)
    elif dataset == 'pubmed':
        get_data = read_pubmed(folds, r=r)
    else:
        raise ValueError(f"Dataset \"{dataset}\" is not recognized.")
    return get_data