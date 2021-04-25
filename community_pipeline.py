import numpy as np
import os, json
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
from models.metrics import dp_link_divergence, recall_at_k, dp_at_k
from preprocess.split_data import split_train_and_test

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)

EMBEDDING_DIM = 100
LR = 1e-3
EPOCHS = 200
LAMBDA = 1
K = 30

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
    nodes = np.loadtxt('preprocess/fb_features_ego_1684.txt')
    edges = np.loadtxt('preprocess/fb_adjacency_1684.txt')
    features = np.concatenate([nodes[:, :147], nodes[:, 149:]], axis = -1)
    attributes = nodes[:, [147, 148]]
    print(attributes.sum() / len(attributes))

    args = type('Args', (object,), {})
    args.train_prop = 0.8
    train_edges, test_edges = split_train_and_test(args, edges)
    use_node = np.any((train_edges != 0), axis = -1)

    features, train_edges = nodes[None, use_node, :], train_edges[None, None, use_node][..., use_node]
    test_edges, attributes = test_edges[use_node][..., use_node], attributes[None, use_node, :]

    pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()

    print("Initial DP diveregence:")
    print(dp_link_divergence(attributes, train_edges).mean())

    np.save('results/community_attributes.npy', attributes[0])

    results = defaultdict(list)

    for n in [1, 2, 4, 8, 16, 32] + list(range(64, 772, 25)):

        COMMUNITY_FAIRNESS = FairCommunityAdditionGraphConv(n)

        #community
        community = FairModel(*features.shape[-2:], COMMUNITY_FAIRNESS, tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu'), reconstruction_model(features.shape[-2], EMBEDDING_DIM))
        community.compile(tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR * LAMBDA), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
        fl, tl = community.fit(features, train_edges, train_edges, attributes, EPOCHS) 
        fair_nodes, fair_edges = community.predict_embeddings([features, train_edges])
        fair_nodes = fair_nodes[0]
        results[f'community_{n}'].extend([*tl, *fl])
        results[f'community_{n}'].append(recall_at_k(fair_nodes, test_edges, k=K))
        results[f'community_{n}'].append(dp_at_k(fair_nodes, attributes[0], k=K))

    results = {k: [float(v) for v in results[k]] for k in results}

    with open('results/community_results.json', 'w') as fp:
        json.dump(results, fp, indent=True)
    print(results)

if __name__ == '__main__':
    main()