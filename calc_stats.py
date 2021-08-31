import numpy as np
import networkx as nx
from preprocess.read_data import read_data

def main():
    for dataset in ['citeseer', 'cora', 'facebook', 'pubmed', 'bail', 'german']:
        features, edges, attributes = read_data(dataset, 0)()
        graph = nx.convert_matrix.from_numpy_matrix(edges - np.eye(len(edges)))
        print(dataset, ':', nx.algorithms.cluster.average_clustering(graph))


if __name__ == '__main__':
    main()