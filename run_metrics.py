import numpy as np
import json, argparse
from glob import glob
from collections import defaultdict

from models.metrics import *
from preprocess.read_data import read_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [bail, cora, citeseer, facebook, pubmed].')
    parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds for k-fold cross validation.')
    return parser.parse_args()

def main():
    
    args = parse_args()
    get_data = read_data(args.dataset, args.folds)

    results = defaultdict(list)
    for model in ['base', 'gfo', 'cfo_10', 'cfo_100', 'few']:
        files = glob(f'./results/{args.dataset}/embeddings/{model.replace("_", "")}*')
        for i, fname in enumerate(files):
            f_results = {}
            embedding = np.load(f'./results/{args.dataset}/embeddings/{fname}')
            
            #get metrics as necessary
            
            results[model].append(f_results)
    
    with open(f'./results/{args.dataset}/results2.json') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    main()