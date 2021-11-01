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
from models.losses import dp_link_divergence_loss, dp_link_entropy_loss, build_reconstruction_loss
from models.metrics import dp_link_divergence, recall_at_k, dp_at_k
from preprocess.split_data import split_train_and_test
from preprocess.read_data import read_data
from main import kfold_fair_model, reconstruction_model

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
    if args.embedding_dim2 == 0:
        args.embedding_dim2 = args.embedding_dim
    features, _, attributes = read_data(args.dataset, args.folds)
 
    results = defaultdict(dict)

    fold_names = []
    for i in range(args.folds):
        fold_names.append((f'./data/{args.dataset}/folds/fold{i}_train.npy',
                           f'./data/{args.dataset}/folds/fold{i}_test.npy'))

    for l in [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        
        print(f'Model gfo_lambda{l}: START')
        args.lambda_param = l
        results[f'gfo_lambda{l}']= kfold_fair_model(features, fold_names, attributes, GFOGraphConv, args)
        print(f'Model gfo_lambda{l}: FINISHED')

    with open(f'results/{args.dataset}/lambda_results.json', 'w') as fp:
        json.dump(results, fp, indent=True, default=to_serializable)

if __name__ == '__main__':
    main()