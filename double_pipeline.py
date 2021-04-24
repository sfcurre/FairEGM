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
from models.losses import dp_link_divergence_loss, dp_link_entropy_loss, build_reconstruction_loss
from models.metrics import dp_link_divergence, recall_at_k, dp_at_k_div, dp_at_k_dif
from preprocess.split_data import split_train_and_test

import tensorflow as tf

dp_at_k = dp_at_k_dif

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)

EMBEDDING_DIM = 100
LR = 1e-4
EPOCHS = 300
LAMBDA = 10
K = 20
TARGETED_FAIRNESS = FairTargetedAdditionGraphConv()
COMMUNITY_FAIRNESS = FairCommunityAdditionGraphConv(10)
SPARSE_FAIRNESS = FairReductionGraphConv()


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

def main():
    nodes = np.loadtxt('preprocess/fb_features.txt')
    edges = np.loadtxt('preprocess/fb_adjacency.txt')
    features = np.concatenate([nodes[:, :77], nodes[:, 79:]], axis = -1)
    attributes = nodes[:, 77:79]

    args = type('Args', (object,), {})
    args.train_prop = 0.8
    train_edges, test_edges = split_train_and_test(args, edges)
    use_node = np.any((train_edges != 0), axis = -1)

    features, train_edges = nodes[None, use_node, :], train_edges[None, None, use_node][..., use_node]
    test_edges, attributes = test_edges[use_node][..., use_node], attributes[None, use_node, :]

    pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()

    print("Initial DP diveregence:")
    print(dp_link_divergence(attributes, train_edges).mean())

    def dp_metric(y_true, y_pred):
        return dp_link_divergence_loss(attributes.astype(np.float32), y_pred)

    results = defaultdict(list)

    dp_total = 0
    for indices in train_edges[0, 0, ...]:
        distro = attributes[0, indices == 1].sum(axis = 0) / attributes[0].sum(axis = 0)
        dp_total += abs(distro[1] - distro[0])
    results['observed'].append(dp_total / train_edges.shape[-1])

    #base
    base, base_embedding = base_model(*features.shape[-2:])
    base.summary()
    base.compile(tf.keras.optimizers.Adam(LR), build_reconstruction_loss(pos_weight), [dp_metric])
    base.fit([features, train_edges], train_edges, epochs = EPOCHS)
    #not actually fair for base
    fair_nodes, fair_edges = base_embedding.predict([features, train_edges])
    fair_nodes = fair_nodes[0]
    results['base'].extend(base.evaluate([features, train_edges], train_edges))
    results['base'].append(recall_at_k(fair_nodes, test_edges, k=K))
    results['base'].append(dp_at_k(fair_nodes, attributes[0], k=K))

    #targeted
    targeted = FairModel(*features.shape[-2:], TARGETED_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(features.shape[-2], EMBEDDING_DIM))
    targeted.compile(tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR * LAMBDA), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
    fl, tl = targeted.fit(features, train_edges, train_edges, attributes, EPOCHS)
    fair_nodes, fair_edges = targeted.predict_embeddings([features, train_edges])
    fair_nodes = fair_nodes[0]
    results['targeted'].extend([*tl, *fl])
    results['targeted'].append(recall_at_k(fair_nodes, test_edges, k=K))
    results['targeted'].append(dp_at_k(fair_nodes, attributes[0], k=K))    

    #community
    community = FairModel(*features.shape[-2:], COMMUNITY_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(features.shape[-2], EMBEDDING_DIM))
    community.compile(tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR * LAMBDA), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
    fl, tl = community.fit(features, train_edges, train_edges, attributes, EPOCHS) 
    fair_nodes, fair_edges = community.predict_embeddings([features, train_edges])
    fair_nodes = fair_nodes[0]
    results['community'].extend([*tl, *fl])
    results['community'].append(recall_at_k(fair_nodes, test_edges, k=K))
    results['community'].append(dp_at_k(fair_nodes, attributes[0], k=K))

    #reduction
    sparse = FairModel(*features.shape[-2:], SPARSE_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(features.shape[-2], EMBEDDING_DIM))
    sparse.compile(tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR * LAMBDA), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
    fl, tl = sparse.fit(features, train_edges, train_edges, attributes, EPOCHS)
    fair_nodes, fair_edges = sparse.predict_embeddings([features, train_edges])
    fair_nodes = fair_nodes[0]
    results['reduction'].extend([*tl, *fl])
    results['reduction'].append(recall_at_k(fair_nodes, test_edges, k=K))
    results['reduction'].append(dp_at_k(fair_nodes, attributes[0], k=K))

    print(results)

if __name__ == '__main__':
    main()