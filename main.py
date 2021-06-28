import numpy as np
import os, json
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict
from functools import singledispatch
import argparse

from layers.graph_cnn import GraphCNN
from layers.gfo_graph_conv import GFOGraphConv
from layers.cfo_graph_conv import CFOGraphConv
from layers.fw_graph_conv import FWGraphConv
from layers.link_prediction import LinkPrediction
from layers.link_reconstruction import LinkReconstruction
from models.fair_model import FairModel
from models.losses import dp_link_divergence_loss, dp_link_entropy_loss, build_reconstruction_loss
from models.metrics import dp_link_divergence, recall_at_k, dp_at_k
from preprocess.read_data import read_data

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)
tf.random.set_seed(5429)

TARGETED_FAIRNESS = lambda: GFOGraphConv()
COMMUNITY_FAIRNESS_10 = lambda: CFOGraphConv(10)
COMMUNITY_FAIRNESS_100 = lambda: CFOGraphConv(100)
SPARSE_FAIRNESS = lambda: FWGraphConv()

@singledispatch
def to_serializable(val):
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    return np.float64(val)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    parser.add_argument('-d', '--embedding-dim', type=int, default=100, help="The graph embedding dimension.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help="Learning rate for the embedding model.")
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs for the embedding model.')
    parser.add_argument('-k', '--top-k', type=int, default=[10], nargs='+', help='Number for K for Recall@K and DP@K metrics.')
    parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds for k-fold cross validation.')
    parser.add_argument('--lambda-param', type=float, default=1, help='The learning rate multiplier for the fair loss.')
    return parser.parse_args()

def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(axis=1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

def reconstruction_model(num_nodes, num_features):
    # num_features == embedding_dim
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((num_nodes, num_nodes))
    conv_nodes, conv_edges = GraphCNN(num_features, activation='linear')([nodes, edges])
    output = LinkReconstruction()(conv_nodes)
    return tf.keras.models.Model([nodes, edges], output)

def base_model(num_nodes, num_features, embedding_dim):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(embedding_dim, activation='relu')([nodes, edges])
    conv_nodes, conv_edges = GraphCNN(embedding_dim, activation='linear')([conv_nodes, conv_edges])
    output = LinkReconstruction()(conv_nodes)
    return tf.keras.models.Model([nodes, edges], output), tf.keras.models.Model([nodes, edges], [conv_nodes, conv_edges])

def kfold_base_model(all_features, fold_generator, all_attributes, args):
    results = []
    for i, (train_edges, test_edges) in enumerate(fold_generator):
        rdict = {}

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[None, use_node, :], train_edges[None, use_node][..., use_node]
        test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]
        norm_edges = preprocess_graph(train_edges[0])[None,...]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        
        def dp_metric(y_true, y_pred):
            return dp_link_divergence_loss(attributes.astype(np.float32), y_pred)

        base, base_embedding = base_model(*features.shape[-2:], args.embedding_dim)
        base.compile(tf.keras.optimizers.Adam(args.learning_rate), build_reconstruction_loss(pos_weight), [dp_metric])
        history = base.fit([features, norm_edges], train_edges, epochs = args.epochs, verbose = 2).history
        history['task loss'] = history.pop('loss')
        history['fair loss'] = history.pop('dp_metric')
        rdict['history'] = history
        #not actually fair for base
        fair_nodes, _ = base_embedding.predict([features, norm_edges])
        fair_nodes = fair_nodes[0]
        recon_loss, dp_loss = base.evaluate([features, norm_edges], train_edges)
        rdict['reconstruction loss'] = recon_loss
        rdict['link divergence'] = dp_loss
        for k in args.top_k:
            rdict[f'recall@{k}'] = recall_at_k(fair_nodes, test_edges, k=k)
            rdict[f'dp@{k}'] = dp_at_k(fair_nodes, attributes[0], k=k)
        results.append(rdict)
    return results

def kfold_fair_model(all_features, fold_generator, all_attributes, layer_constructor, args):
    results = []
    for i, (train_edges, test_edges) in enumerate(fold_generator):
        rdict = {}

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[None, use_node, :], train_edges[None, use_node][..., use_node]
        test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]
        norm_edges = preprocess_graph(train_edges[0])[None,...]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        
        model = FairModel(*features.shape[-2:], layer_constructor(), tf.keras.layers.Dense(args.embedding_dim, activation='relu'), reconstruction_model(features.shape[-2], args.embedding_dim))
        model.compile(tf.keras.optimizers.Adam(args.learning_rate), tf.keras.optimizers.Adam(args.learning_rate * args.lambda_param), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
        history = model.fit(features, norm_edges, train_edges, attributes, args.epochs, verbose=0)
        rdict['history'] = history
        fair_nodes, fair_edges = model.predict_embeddings([features, norm_edges])
        fair_nodes = fair_nodes[0]
        recon_loss, dp_loss = model.evaluate(features, norm_edges, train_edges, attributes)
        rdict['reconstruction loss'] = recon_loss
        rdict['link divergence'] = dp_loss
        for k in args.top_k:
            rdict[f'recall@{k}'] = recall_at_k(fair_nodes, test_edges, k=k)
            rdict[f'dp@{k}'] = dp_at_k(fair_nodes, attributes[0], k=k)
        print(f'Model {i+1}: [{rdict["reconstruction loss"]},{rdict["link divergence"]}]')
        results.append(rdict)
    return results

def main():
    args = parse_args()
    get_data = read_data(args.dataset, args.folds)

    results = {}
    data=get_data()
    results['base'] = kfold_base_model(*data, args)
    data=get_data()
    results['gfo'] = kfold_fair_model(*data, TARGETED_FAIRNESS, args)
    data=get_data()
    results['cfo_10'] = kfold_fair_model(*data, COMMUNITY_FAIRNESS_10, args)
    data=get_data()
    results['cfo_100'] = kfold_fair_model(*data, COMMUNITY_FAIRNESS_100, args)
    data=get_data()
    results['fw'] = kfold_fair_model(*data, SPARSE_FAIRNESS, args)

    with open(f'./results/{args.dataset}/results.json', 'w') as fp:
        json.dump(results, fp, indent=True, default=to_serializable)


if __name__ == '__main__':
    main()