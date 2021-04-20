import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict

from layers.graph_cnn import GraphCNN
from layers.targeted_fair_graph_cnn import FairTargetedAdditionGraphConv
from layers.community_fair_graph_cnn import FairCommunityAdditionGraphConv
from layers.sparse_fair_graph_cnn import FairReductionGraphConv
from layers.link_prediction import LinkPrediction
from layers.link_reconstruction import LinkReconstruction
from models.fair_model import FairModel
from models.fair_losses import dp_link_divergence_loss
from models.fair_metrics import dp_link_divergence

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)

TARGETED_FAIRNESS = FairTargetedAdditionGraphConv()
COMMUNITY_FAIRNESS = FairCommunityAdditionGraphConv(10)
SPARSE_FAIRNESS = FairReductionGraphConv()

def link_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))
    output = LinkPrediction(100, activation='sigmoid')(nodes)
    return tf.keras.models.Model([nodes, edges], output)

def base_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(300, activation='relu')([nodes, edges])
    output = link_model(num_nodes, 300)([conv_nodes, conv_edges])
    return tf.keras.models.Model([nodes, edges], output)

def main():
    nodes = np.loadtxt('preprocess/fb_features.txt')
    edges = np.loadtxt('preprocess/fb_adjacency.txt')
    features = np.concatenate([nodes[:, :77], nodes[:, 79:]], axis = -1)
    attributes = nodes[:, 77:79]

    targets = (edges * np.random.rand(*edges.shape) > 0.9).astype(float)
    edges -= targets
    targets = targets + edges

    features, edges, attributes, targets = nodes[None, ...], edges[None, None, ...], attributes[None, ...], targets[None, ...]

    print("Initial DP diveregence:")
    print(dp_link_divergence(attributes, targets).mean())

    def dp_metric(y_true, y_pred):
        return dp_link_divergence_loss(attributes.astype(np.float32), y_pred)

    def accuracy_metric(y_true, y_pred):
        return tf.keras.metrics.binary_accuracy(targets.astype(np.float32), y_pred)   

    results = defaultdict(list)

    #base
    base = base_model(*nodes.shape[-2:])
    base.compile(tf.keras.optimizers.Adam(1e-4), 'binary_crossentropy', [accuracy_metric, dp_metric])
    base.fit([features, edges], targets, epochs = 100)
    output = base.predict([features, edges])
    results['base'].append(np.sum(targets == (output > 0.5)) / output.size)
    results['base'].append(dp_link_divergence(attributes, output).mean())
    del base

    #targeted
    targeted = FairModel(*nodes.shape[-2:], TARGETED_FAIRNESS, tf.keras.layers.Dense(100, activation='relu'), link_model(len(nodes), 100))
    targeted.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4), tf.keras.losses.binary_crossentropy, dp_link_divergence_loss, [accuracy_metric], [dp_metric])
    targeted.fit(features, edges, targets, attributes, 100)
    output = targeted.predict([features, edges])
    results['targeted'].append(np.sum(targets == (output > 0.5)) / output.size)
    results['targeted'].append(dp_link_divergence(attributes, output).mean())
    del targeted

    #community
    community = FairModel(*nodes.shape[-2:], COMMUNITY_FAIRNESS, tf.keras.layers.Dense(100, activation='relu'), link_model(len(nodes), 100))
    community.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4), tf.keras.losses.binary_crossentropy, dp_link_divergence_loss, [accuracy_metric], [dp_metric])
    community.fit(features, edges, targets, attributes, 100)
    output = community.predict([features, edges])
    results['community'].append(np.sum(targets == (output > 0.5)) / output.size)
    results['community'].append(dp_link_divergence(attributes, output).mean())
    del community

    #reduction
    sparse = FairModel(*nodes.shape[-2:], SPARSE_FAIRNESS, tf.keras.layers.Dense(100, activation='relu'), link_model(len(nodes), 100))
    sparse.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4), tf.keras.losses.binary_crossentropy, dp_link_divergence_loss, [accuracy_metric], [dp_metric])
    sparse.fit(features, edges, targets, attributes, 100)
    output = sparse.predict([features, edges])
    results['sparse'].append(np.sum(targets == (output > 0.5)) / output.size)
    results['sparse'].append(dp_link_divergence(attributes, output).mean())
    del sparse

    print(results)

if __name__ == '__main__':
    main()