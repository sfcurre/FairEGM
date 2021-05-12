import numpy as np
import os, json
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict
import argparse

from layers.graph_cnn import GraphCNN
from layers.targeted_fair_graph_cnn import FairTargetedAdditionGraphConv
from layers.community_fair_graph_cnn import FairCommunityAdditionGraphConv
from layers.sparse_fair_graph_cnn import FairReductionGraphConv
from layers.link_prediction import LinkPrediction
from layers.link_reconstruction import LinkReconstruction
from models.fair_model import FairModel
from models.losses import dp_link_divergence_loss, dp_link_entropy_loss, build_reconstruction_loss
from models.metrics import dp_link_divergence, recall_at_k, dp_at_k
from preprocess.read_data import *

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)
tf.random.set_seed(5429)

TARGETED_FAIRNESS = FairTargetedAdditionGraphConv()
COMMUNITY_FAIRNESS_10 = FairCommunityAdditionGraphConv(10)
COMMUNITY_FAIRNESS_100 = FairCommunityAdditionGraphConv(100)
SPARSE_FAIRNESS = FairReductionGraphConv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    parser.add_argument('-d', '--embedding-dim', type=int, default=100, help="The graph embedding dimension.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help="Learning rate for the embedding model.")
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs for the embedding model.')
    parser.add_argument('-k', '--top-k', type=int, default=10, help='Number for K for Recall@K and DP@K metrics.')
    parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds for k-fold cross validation.')
    parser.add_argument('--lambda', type=float, default=1, help='The learning rate multiplier for the fair loss.')
    return parser.parse_args()

def reconstruction_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))
    output = LinkReconstruction()(nodes)
    return tf.keras.models.Model([nodes, edges], output)

def base_model(num_nodes, num_features, embedding_dim):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(embedding_dim, activation='relu')([nodes, edges])
    output = reconstruction_model(num_nodes, embedding_dim)([conv_nodes, conv_edges])
    return tf.keras.models.Model([nodes, edges], output), tf.keras.models.Model([nodes, edges], [conv_nodes, conv_edges])

def kfold_base_model(all_features, train_folds, test_folds, all_attributes, args):
    results = []
    for i in range(args.folds):
        rdict = {}
        train_edges = train_folds[i]
        test_edges = test_folds[i]

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[None, use_node, :], train_edges[None, None, use_node][..., use_node]
        test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        
        def dp_metric(y_true, y_pred):
            return dp_link_divergence_loss(attributes.astype(np.float32), y_pred)

        base, base_embedding = base_model(*features.shape[-2:], args.embedding_dim)
        base.compile(tf.keras.optimizers.Adam(args.learning_rate), build_reconstruction_loss(pos_weight), [dp_metric])
        history = base.fit([features, train_edges], train_edges, epochs = args.epochs).history
        rdict['history'] = {k: [float(v) for v in history[k]] for k in history}
        #not actually fair for base
        fair_nodes, _ = base_embedding.predict([features, train_edges])
        fair_nodes = fair_nodes[0]
        recon_loss, dp_loss = base.evaluate([features, train_edges], train_edges)
        rdict['reconstruction loss'] = recon_loss
        rdict['link divergence'] = dp_loss
        rdict['recall@k'] = recall_at_k(fair_nodes, test_edges, k=args.top_k)
        rdict['dp@k'] = dp_at_k(fair_nodes, attributes[0], k=args.top_k)
        results.append(rdict)
    return results

def kfold_fair_model(all_features, train_folds, test_folds, all_attributes, layer_constructor, args):
    results = []
    for i in range(args.folds):
        rdict = {}
        train_edges = train_folds[i]
        test_edges = test_folds[i]

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[None, use_node, :], train_edges[None, None, use_node][..., use_node]
        test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        
        model = FairModel(*features.shape[-2:], layer_constructor(), tf.keras.layers.Dense(args.embedding_dim, activation='relu'), reconstruction_model(features.shape[-2], args.embedding_dim))
        targeted.compile(tf.keras.optimizers.Adam(args.learning_rate), tf.keras.optimizers.Adam(args.learning_rate * args.lambda), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
        history = targeted.fit(features, train_edges, train_edges, attributes, args.epochs)
        rdict['history'] = {k: [float(v) for v in history[k]] for k in history}
        fair_nodes, fair_edges = targeted.predict_embeddings([features, train_edges])
        fair_nodes = fair_nodes[0]
        rdict['reconstruction loss'] = history['tl'][-1]
        rdict['link divergence'] = history['fl'][-1]
        rdict['recall@k'] = recall_at_k(fair_nodes, test_edges, k=args.top_k)
        rdict['dp@k'] = dp_at_k(fair_nodes, attributes[0], k=args.top_k)
        results.append(rdict)
    return results

def main():
    args = parse_args()
    if args.dataset == 'citeseer':
        data = read_citeseer(args.folds)
    elif args.dataset == 'cora':
        data = read_cora(args.folds)
    elif args.dataset == 'facebook':
        data = read_facebook(args.folds)
    else:
        raise argparse.ArgumentError(f"Dataset \"{args.dataset}\" is not recognized.")
    


if __name__ == '__main__':
    main()