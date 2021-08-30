import numpy as np
import os, json, argparse
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict
from functools import singledispatch

from layers.graph_cnn import GraphCNN
from layers.cfo_graph_conv import CFOGraphConv
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
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    parser.add_argument('-d', '--embedding-dim', type=int, default=100, help="The graph embedding dimension.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help="Learning rate for the embedding model.")
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs for the embedding model.')
    parser.add_argument('-k', '--top-k', type=int, default=[10], nargs='+', help='Number for K for Recall@K and DP@K metrics.')
    parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds for k-fold cross validation. ONLY THE FIRST FOLD IS USED.')
    parser.add_argument('--lambda-param', type=float, default=1, help='The learning rate multiplier for the fair loss.')
    return parser.parse_args()

def main():

    args = parse_args()
    get_data = read_data(args.dataset, args.folds)
    results = defaultdict(dict)

    for n in range(1, 30, 3):#[1, 2, 4, 8, 16] + list(range(30, 360, 10)):

        data = get_data()
        COMMUNITY_FAIRNESS = lambda: CFOGraphConv(n)

        print(f'Model cfo_{n}: START')
        results[f'cfo_{n}']= kfold_fair_model(*data, COMMUNITY_FAIRNESS, args)
        print(f'Model cfo_{n}: FINISHED')

    with open(f'results/{args.dataset}/community_results.json', 'w') as fp:
        json.dump(results, fp, indent=True, default=to_serializable)

if __name__ == '__main__':
    main()