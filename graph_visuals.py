import numpy as np, pandas as pd
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from layers.gfo_graph_conv import GFOGraphConv
from models.losses import build_reconstruction_loss, dp_link_divergence_loss
from models.fair_model import FairModel
from preprocess.read_data import *
from main import reconstruction_model, base_model

from pyclustertend import compute_ordered_dissimilarity_matrix, vat, ivat
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

import tensorflow as tf

sns.set_theme()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    parser.add_argument('-d', '--embedding-dim', type=int, default=100, help="The graph embedding dimension.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help="Learning rate for the embedding model.")
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs for the embedding model.')
    parser.add_argument('--lambda-param', type=float, default=1, help='The learning rate multiplier for the fair loss.')
    return parser.parse_args()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def umap_graph(adj, attributes, seed=1, **kwargs):
    np.random.seed(seed)
    reducer = umap.UMAP(**kwargs, transform_seed=seed)
    umap_embeds = reducer.fit_transform(adj)
    plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1],
    c=[sns.color_palette()[x] for x in np.argmax(attributes, axis = -1)])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of Graph embeddings', fontsize=24)

def arrange_nodes_by_attributes(nodes, attributes):
    attr_scalar = attributes.argmax(axis=-1)
    blocks = []
    for i in range(attributes.shape[-1]):
        blocks.append(nodes[attr_scalar == i])
    return np.concatenate(blocks, axis = 0)

#=====================================================================
def main():
    args = parse_args()
    if args.dataset == 'citeseer':
        data = read_citeseer(0)
    elif args.dataset == 'cora':
        data = read_cora(0)
    elif args.dataset == 'credit':
        data = read_credit(0)
    elif args.dataset == 'facebook':
        data = read_facebook(0)
    elif args.dataset == 'pubmed':
        data = read_pubmed(0)
    else:
        raise ValueError(f"Dataset \"{args.dataset}\" is not recognized.")

    features, train_edges, _, attributes = data

    use_node = np.any((train_edges != 0), axis = -1)

    train_edges = train_edges[None, use_node][..., use_node]
    features, attributes = features[None, use_node, :], attributes[None, use_node, :]

    pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()

    #Base model
    base, base_embedding = base_model(*features.shape[-2:], args.embedding_dim)
    base.compile(tf.keras.optimizers.Adam(args.learning_rate), build_reconstruction_loss(pos_weight), [])
    base.fit([features, train_edges], train_edges, epochs = args.epochs, verbose = 0)
    base_nodes, base_edges = base_embedding.predict([features, train_edges])
    base_nodes = base_nodes[0]

    #Fair Model
    model = FairModel(*features.shape[-2:], GFOGraphConv(), tf.keras.layers.Dense(args.embedding_dim, activation='relu'), reconstruction_model(features.shape[-2], args.embedding_dim))
    model.compile(tf.keras.optimizers.Adam(args.learning_rate), tf.keras.optimizers.Adam(args.learning_rate * args.lambda_param), build_reconstruction_loss(pos_weight), dp_link_divergence_loss)
    model.fit(features, train_edges, train_edges, attributes, args.epochs)
    fair_nodes, fair_edges = model.predict_embeddings([features, train_edges])
    fair_nodes = fair_nodes[0]

    attributes = attributes[0]
    base_nodes = arrange_nodes_by_attributes(base_nodes, attributes)
    fair_nodes = arrange_nodes_by_attributes(fair_nodes, attributes)
    
    pca = PCA(2)
    pca_base = pca.fit_transform(base_nodes)
    pca_fair = pca.fit_transform(fair_nodes)

    plt.scatter(pca_base[:, 0], pca_base[:, 1], c=[sns.color_palette()[x] for x in np.argmax(attributes, axis = -1)])
    plt.savefig(f'./visuals/images/{args.dataset}_pca_base.png')
    plt.show()
    
    plt.scatter(pca_fair[:, 0], pca_fair[:, 1], c=[sns.color_palette()[x] for x in np.argmax(attributes, axis = -1)])
    plt.savefig(f'./visuals/images/{args.dataset}_pca_fair.png')
    plt.show()

    # base_reconstruction = sigmoid(base_nodes @ base_nodes.T)
    # fair_reconstruction = sigmoid(fair_nodes @ fair_nodes.T)
    
    # plt.matshow(base_reconstruction, cmap='binary', vmin=0)
    # plt.show()
    # plt.matshow(fair_reconstruction, cmap='binary', vmin=0)
    # plt.show()

    # vat(base_nodes)
    # plt.show()
    # ivat(base_nodes)
    # plt.show()

    # vat(fair_nodes)
    # plt.show()
    # ivat(fair_nodes)
    # plt.show()

    #umap_kwargs = dict(n_neighbors = 15, min_dist = 0.15)

    #original
    #umap_graph(train_edges[0, 0], attributes, **umap_kwargs)
    #plt.show()

    #base
    #umap_graph(base_reconstruction, attributes, **umap_kwargs)
    #plt.show()

    #fair
    #umap_graph(reconstruction, attributes, **umap_kwargs)
    #plt.show()

if __name__ == '__main__':
    main()