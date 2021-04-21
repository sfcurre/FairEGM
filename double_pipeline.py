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

EMBEDDING_DIM = 100
TARGETED_FAIRNESS = FairTargetedAdditionGraphConv()
COMMUNITY_FAIRNESS = FairCommunityAdditionGraphConv(10)
SPARSE_FAIRNESS = FairReductionGraphConv()

def construct_datasets(nodes, edges):
    pass

def reconstruction_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))
    output = LinkReconstruction()(nodes)
    return tf.keras.models.Model([nodes, edges], output)

def base_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(EMBEDDING_DIM, activation='relu')([nodes, edges])
    output = reconstruction_model(num_nodes, EMBEDDING_DIM)([conv_nodes, conv_edges])
    return tf.keras.models.Model([nodes, edges], output), tf.keras.models.Model([nodes, edges], [conv_nodes, conv_edges])

def link_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))
    output = LinkPrediction(100, activation='sigmoid')(nodes)
    return tf.keras.models.Model([nodes, edges], output)

def main():
    nodes = np.loadtxt('preprocess/fb_features.txt')
    edges = np.loadtxt('preprocess/fb_adjacency.txt')
    features = np.concatenate([nodes[:, :77], nodes[:, 79:]], axis = -1)
    attributes = nodes[:, 77:79]

    targets = (edges * np.random.rand(*edges.shape) > 0.8).astype(float)
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
    base, base_embedding = base_model(*nodes.shape[-2:])
    base.compile(tf.keras.optimizers.Adam(1e-4), 'binary_crossentropy', [accuracy_metric, dp_metric])
    base.fit([features, edges], targets, epochs = 100)
    #not actually fair for base
    fair_nodes, fair_edges = base_embedding.predict([features, edges])
    

    #targeted
    targeted = FairModel(*nodes.shape[-2:], TARGETED_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(len(nodes), EMBEDDING_DIM))
    targeted.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4), tf.keras.losses.binary_crossentropy, dp_link_divergence_loss, [accuracy_metric], [dp_metric])
    targeted.fit(features, edges, targets, attributes, 100)
    fair_nodes, fair_edges = targeted.predict_embeddings([features, edges])
    

    #community
    community = FairModel(*nodes.shape[-2:], COMMUNITY_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(len(nodes), EMBEDDING_DIM))
    community.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4), tf.keras.losses.binary_crossentropy, dp_link_divergence_loss, [accuracy_metric], [dp_metric])
    community.fit(features, edges, targets, attributes, 100) 
    fair_nodes, fair_edges = community.predict_embeddings([features, edges])
    

    #reduction
    sparse = FairModel(*nodes.shape[-2:], SPARSE_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(len(nodes), EMBEDDING_DIM))
    sparse.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4), tf.keras.losses.binary_crossentropy, dp_link_divergence_loss, [accuracy_metric], [dp_metric])
    sparse.fit(features, edges, targets, attributes, 100)
    fair_nodes, fair_edges = sparse.predict_embeddings([features, edges])
    


if __name__ == '__main__':
    main()