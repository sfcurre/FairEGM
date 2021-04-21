from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import networkx as nx

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='fb_adjacency.txt',
                        help='Input graph file')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--train-prop', default=0.8, required=False, type=float,
                        help='Proportion of training edges. Default is 0.8.')
    parser.add_argument('--basic-embed', required=False, default='netmf',
                        choices=['deepwalk', 'grarep', 'netmf', 'sdne'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--refine-type', required=False, default='MD-gcn',
                        choices=['MD-gcn', 'MD-dumb', 'MD-gs'],
                        help='The method for refining embeddings.')
    # parser.add_argument('--coarsen-level', default=2, type=int,
    #                     help='MAX number of levels of coarsening.')
    # parser.add_argument('--coarsen-to', default=500, type=int,
    #                     help='MAX number of nodes in the coarest graph.')
    parser.add_argument('--coarsen-k', default=2, type=int,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--workers', default=28, type=int,
                        help='Number of workers.')
    parser.add_argument('--batch-size', default=1e5, type=int,
                        help='Batch size for applying the refinement model')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]
    # Consider increasing self-weight a little bit if coarsen-level is high.
    args = parser.parse_args()
    return args

def is_in_train(train, node):
    return np.sum(train[node, :]) > 0

def split_train_and_test(args, data):
    print(np.sum(data))
    # preprocess
    data[np.tril_indices_from(data)] = 0  # remove all edges (u, v) such that u >= v, including self-loops and duplicate edges
    idx_i, idx_j = np.where(data > 0)  # construct edgelist
    node_num = data.shape[0]
    edge_num = len(idx_i)

    # randomly select (1-args.train_prop) positive test edges
    permu = np.random.permutation(edge_num)
    train_mtx = np.zeros_like(data, dtype=np.int32)
    train_mtx[idx_i[permu[:int(edge_num * args.train_prop)]], idx_j[permu[:int(edge_num * args.train_prop)]]] = 1
    test_mtx = np.zeros_like(data, dtype=np.int32)
    test_mtx[idx_i[permu[int(edge_num * args.train_prop):]], idx_j[permu[int(edge_num * args.train_prop):]]] = 1
    assert np.array_equal(train_mtx + test_mtx, data)
    print(np.sum(train_mtx) * 2, np.sum(test_mtx) * 2, np.sum(train_mtx + test_mtx) * 2)

    # find the largest (weakly) connected component of the training set
    train_nx = nx.convert_matrix.from_numpy_array(train_mtx)
    largest_cc = max(nx.connected_components(train_nx), key=len)
    for i in range(node_num):
        if i not in largest_cc:
            train_mtx[i, :] = 0
            train_mtx[:, i] = 0
    train_mtx += train_mtx.T
    print(np.sum(train_mtx))

    # remove test nodes not in the training set
    idx_i, idx_j = np.where(test_mtx > 0)
    edge_num_test = len(idx_i)
    for e in range(edge_num_test):
        i, j = idx_i[e], idx_j[e]
        if test_mtx[i, j] == 0:
            continue
        if not is_in_train(train_mtx, i):
            test_mtx[i, :] = 0
            test_mtx[:, i] = 0
        if not is_in_train(train_mtx, j):
            test_mtx[j, :] = 0
            test_mtx[:, j] = 0
    test_mtx += test_mtx.T  # symmetric
    
    print(np.sum(train_mtx), np.sum(test_mtx), np.sum(train_mtx + test_mtx))
    return train_mtx, test_mtx



if __name__ == "__main__":
    args = parse_args()

    data = np.loadtxt(args.data)
    train_mtx, test_mtx = split_train_and_test(args, data)

    np.savetxt("train_mtx.txt", train_mtx, fmt='%1d')
    np.savetxt("test_mtx.txt", test_mtx, fmt='%1d')
