import numpy as np
import json, argparse
from glob import glob
from collections import defaultdict

from models.metrics import *
from preprocess.read_data import read_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [bail, cora, citeseer, facebook, pubmed].')
    parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds for k-fold cross validation.')
    return parser.parse_args()

def main():
    
    args = parse_args()
    all_features, _, all_attributes = read_data(args.dataset, args.folds)

    fold_names = []
    for i in range(args.folds):
        fold_names.append((f'./data/{args.dataset}/folds/fold{i}_train.npy',
                           f'./data/{args.dataset}/folds/fold{i}_test.npy'))

    results = defaultdict(list)
    for model in ['base', 'gfo', 'cfo_10', 'cfo_100', 'few']:
        np.random.seed(5429)

        files = glob(f'./results/{args.dataset}/embeddings/{model.replace("_", "")}*')
        for fname, (train_edges, test_edges) in zip(files, fold_names):
            f_results = {}
            embeddings = np.load(fname)
            train_edges = np.load(train_edges)
            test_edges = np.load(test_edges)

            use_node = np.any((train_edges != 0), axis = -1)
            features, train_edges = all_features[None, use_node, :], train_edges[use_node][..., use_node]
            test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]

            #get positive and negative test edges            
            node_indices = np.indices(test_edges.shape)

            pos_train = node_indices[:, np.triu(train_edges) == 1].T
            pos_test = node_indices[:, np.triu(test_edges) == 1].T
            neg_indices = node_indices[:, np.triu((test_edges + train_edges)==0) == 1].T
            neg_indices = neg_indices[np.random.choice(len(neg_indices), size=len(pos_train)+len(pos_test), replace=False)]
            neg_train, neg_test = neg_indices[:int(0.8 * len(neg_indices))], neg_indices[int(0.8 * len(neg_indices)):]

            train_indices = np.concatenate([pos_train, neg_train])
            test_indices = np.concatenate([pos_test, neg_test])

            #get metrics as necessary
            train_embeddings = np.concatenate([embeddings[train_indices[:, 0]], embeddings[train_indices[:, 1]]], axis=-1)
            test_embeddings = np.concatenate([embeddings[test_indices[:, 0]], embeddings[test_indices[:, 1]]], axis=-1)
            train_labels = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
            test_labels = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])

            lr_model = LogisticRegression()
            lr_model.fit(train_embeddings, train_labels)

            train_prob_preds = lr_model.predict_proba(train_embeddings)[:,1]
            test_prob_preds = lr_model.predict_proba(test_embeddings)[:,1]
                
            f_results['train_auc'] = roc_auc_score(train_labels, train_prob_preds)
            f_results['test_auc'] = roc_auc_score(test_labels, test_prob_preds)
            f_results['train_f1'] = f1_score(train_labels, (train_prob_preds > 0.5).astype(int))
            f_results['test_f1'] = f1_score(test_labels, (test_prob_preds > 0.5).astype(int))

            results[model].append(f_results)
    
    with open(f'./results/{args.dataset}/results2.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    main()