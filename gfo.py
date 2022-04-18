import numpy as np
import os, json, argparse
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict
from functools import singledispatch

from layers.graph_cnn import GraphCNN
from layers.gfo_graph_conv import GFOGraphConv
from layers.link_prediction import LinkPrediction
from layers.link_reconstruction import LinkReconstruction
from models.fair_model import FairModel
from preprocess.read_data import read_data
from main import kfold_base_model, kfold_fair_model, reconstruction_model

from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

sns.set_theme()

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)

@singledispatch
def to_serializable(val):
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    return np.float64(val)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', 
                        help='The dataset to train models on, one of [cora, citeseer, facebook].')
    parser.add_argument('-d', '--embedding-dim', type=int, default=100, 
                        help="The graph embedding dimension.")
    parser.add_argument('-d2', '--embedding-dim2', type=int, default=0, 
                        help="The second graph embedding dimension. If 0, use the same embedding dimension as the first.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, 
                        help="Learning rate for the embedding model.")
    parser.add_argument('-e', '--epochs', type=int, default=300, 
                        help='Number of epochs for the embedding model.')
    parser.add_argument('-k', '--top-k', type=int, default=[10], nargs='+', 
                        help='Number for K for Recall@K and DP@K metrics.')
    parser.add_argument('-f', '--folds', type=int, default=5, 
                        help='Number of folds for k-fold cross validation. ONLY THE FIRST FOLD IS USED.')
    parser.add_argument('-Le', '--lambda-epochs', type=float, default=1, 
                        help='The number of epochs for the fair loss.')
    return parser.parse_args()

def main():

    args = parse_args()
    
    features, edge_gen, attributes = read_data(args.dataset, args.folds)
    
    fold_names = []
    for i, (train, test) in enumerate(edge_gen):
        np.save(f'./data/{args.dataset}/folds/fold{i}_train.npy', train)
        np.save(f'./data/{args.dataset}/folds/fold{i}_test.npy', test)
        fold_names.append((f'./data/{args.dataset}/folds/fold{i}_train.npy',
                           f'./data/{args.dataset}/folds/fold{i}_test.npy'))
    
    if args.embedding_dim2 == 0:
        args.embedding_dim2 = args.embedding_dim
 
    results = defaultdict(dict)

    args.d = 32
    args.d2 = 16
    args.learning_rate = 1e-4
    args.lambda_param = 1
    args.vgae=False
    TARGETED_FAIRNESS = lambda: GFOGraphConv()
    results = kfold_fair_model(features, fold_names, attributes, TARGETED_FAIRNESS, args,
    embedding_file=f'./results/{args.dataset}/embeddings/gfo_temp', return_weights=True)
    w = results['fair_weights'][0]
    a = results['attributes']
    f = results['features']
    e = results['edges']
    n = results['norm_edges']
    phi = results['embedding']

    results = kfold_base_model(features, fold_names, attributes, args,
    embedding_file=f'./results/{args.dataset}/embeddings/base_temp', return_weights=True)
    ab = results['attributes']
    fb = results['features']
    eb = results['edges']
    nb = results['norm_edges']
    phib = results['embedding']

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    #pca = PCA()
    pca = TSNE(2)
    data_pca = pca.fit_transform(w)
    norm = TwoSlopeNorm(0)

    axes[0].scatter(data_pca[:, 0], data_pca[:, 1], marker='o', c=a.argmax(axis=1), cmap='tab10', vmax=10)
    axes[0].set_title("GFO bias", fontsize=14)
    #pca = PCA()
    pca = TSNE(2)

    # data_pca = pca.fit_transform(n @ f)
    data_pca = pca.fit_transform(phib)

    axes[1].scatter(data_pca[:, 0], data_pca[:, 1], marker='o', c=a.argmax(axis=1), cmap='tab10', vmax=10)
    axes[1].set_title("Base GAE Embeddings", fontsize=14)
    #pca = PCA()
    pca = TSNE(2)

    data_pca = pca.fit_transform(phi)

    axes[2].scatter(data_pca[:, 0], data_pca[:, 1], marker='o', c=a.argmax(axis=1), cmap='tab10', vmax=10)
    axes[2].set_title("GFO embeddings", fontsize=14)

    fig.suptitle("TSNE visualizations of the GFO bias, Base GAE embeddings, and GFO embeddings for the Cora dataset.", fontsize=16)
    plt.show()

    with open(f'results/{args.dataset}/lambda_results.json', 'w') as fp:
        json.dump(results, fp, indent=True, default=to_serializable)

if __name__ == '__main__':
    main()