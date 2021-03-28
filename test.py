import tensorflow as tf
import numpy as np

from layers.graph_cnn import GraphCNN
from layers.targeted_fair_graph_cnn import FairTargetedAdditionGraphConv
from layers.community_fair_graph_cnn import FairCommunityAdditionGraphConv
from layers.sparse_fair_graph_cnn import FairReductionGraphConv
from layers.link_prediction import LinkPrediction
from models.fair_model import FairModel
from models.fair_losses import fair_link_loss

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TARGETED_FAIRNESS = FairTargetedAdditionGraphConv()
COMMUNITY_FAIRNESS = FairCommunityAdditionGraphConv(1)
SPARSE_FAIRNESS = FairReductionGraphConv()

def link_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))
    output = LinkPrediction(2, 1, activation='sigmoid')(nodes)
    return tf.keras.models.Model([nodes, edges], output)

def base_model(num_nodes, num_features):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((1, num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(4, activation='relu')([nodes, edges])
    output = link_model(num_nodes, 4)([conv_nodes, conv_edges])
    return tf.keras.models.Model([nodes, edges], output)

def main():
    #16 nodes, 8 attributes
    fake_nodes = np.random.rand(1, 16, 8)
    fake_edges = (np.random.rand(1, 1, 16, 16) > 0.5).astype(float)

    fake_targets = (fake_edges * np.random.rand(*fake_edges.shape) > 0.75).astype(float)
    fake_edges -= fake_targets

    fake_attributes = (np.random.rand(1, 1, 16, 1) > 0.5).astype(float)
    fake_attributes = np.concatenate([fake_attributes, -(fake_attributes - 1)], axis = -1)

    #base
    base = base_model(16, 8)
    base.compile('Adam', 'binary_crossentropy', ['categorical_accuracy'])
    base.fit([fake_nodes, fake_edges], fake_targets, epochs = 5)

    #targeted
    targeted = FairModel(16, 8, TARGETED_FAIRNESS, tf.keras.layers.Dense(4, activation='relu'), link_model(16, 4))
    targeted.compile('Adam', 'binary_crossentropy', fair_link_loss, ['categorical_accuracy'])
    targeted.fit(fake_nodes, fake_edges, fake_targets, fake_attributes, 5)

    #community
    community = FairModel(16, 8, COMMUNITY_FAIRNESS, tf.keras.layers.Dense(4, activation='relu'), link_model(16, 4))
    community.compile('Adam', 'binary_crossentropy', fair_link_loss, ['categorical_accuracy'])
    community.fit(fake_nodes, fake_edges, fake_targets, fake_attributes, 5)

    #reduction
    sparse = FairModel(16, 8, SPARSE_FAIRNESS, tf.keras.layers.Dense(4, activation='relu'), link_model(16, 4))
    sparse.compile('Adam', 'binary_crossentropy', fair_link_loss, ['categorical_accuracy'])
    sparse.fit(fake_nodes, fake_edges, fake_targets, fake_attributes, 5)

if __name__ == '__main__':
    main()