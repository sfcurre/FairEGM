from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import networkx as nx
import os


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='fb_adjacency.txt',
                        help='Input graph file')
    parser.add_argument('--fold', default=5, type=int,
                        help='k-fold cross-validation')
    args = parser.parse_args()
    return args


def is_in_train(train, node):
    return np.sum(train[node, :]) > 0


def split_train_and_test(args, data):
    np.random.seed(5429)
    #print(np.sum(data))
    # preprocess
    data[np.tril_indices_from(
        data)] = 0  # remove all edges (u, v) such that u >= v, including self-loops and duplicate edges
    idx_i, idx_j = np.where(data > 0)  # construct edgelist
    node_num = data.shape[0]
    edge_num = len(idx_i)

    # randomly select (1-args.train_prop) positive test edges
    permu = np.random.permutation(edge_num)
    for itr in range(args.fold):
        test_st, test_ed = int(float(edge_num) / args.fold * itr), int(float(edge_num) / args.fold * (itr + 1))
        #print("Iteration: %d / %d, [test_st, test_ed] = [%d, %d]"%(itr, args.fold, test_st, test_ed))

        # initial split
        test_mtx = np.zeros_like(data, dtype=np.int32)
        test_mtx[idx_i[permu[test_st:test_ed]], idx_j[permu[test_st:test_ed]]] = 1
        train_mtx = data - test_mtx
        #print(np.sum(train_mtx) * 2, np.sum(test_mtx) * 2, np.sum(train_mtx + test_mtx) * 2)

        # find the largest (weakly) connected component of the training set
        train_nx = nx.convert_matrix.from_numpy_array(train_mtx)
        largest_cc = max(nx.connected_components(train_nx), key=len)
        for i in range(node_num):
            if i not in largest_cc:
                train_mtx[i, :] = 0
                train_mtx[:, i] = 0
        train_mtx += train_mtx.T
        #print(np.sum(train_mtx))

        # remove test nodes not in the training set
        idx_i_test, idx_j_test = np.where(test_mtx > 0)
        edge_num_test = len(idx_i_test)
        for e in range(edge_num_test):
            i, j = idx_i_test[e], idx_j_test[e]
            if test_mtx[i, j] == 0:
                continue
            if not is_in_train(train_mtx, i):
                test_mtx[i, :] = 0
                test_mtx[:, i] = 0
            if not is_in_train(train_mtx, j):
                test_mtx[j, :] = 0
                test_mtx[:, j] = 0
        test_mtx += test_mtx.T  # symmetric

        #print(np.sum(train_mtx), np.sum(test_mtx), np.sum(train_mtx + test_mtx))
        yield train_mtx, test_mtx


if __name__ == "__main__":
    args = parse_args()

    data = np.loadtxt(args.data)
    matrix_gen = split_train_and_test(args, data)

    folder_name = "%d-fold data" % args.fold
    os.makedirs(folder_name, exist_ok=True)

    for i, (train_matrix, test_matrix) in enumerate(matrix_gen):
        np.savetxt("%s/train_mtx-%d.txt" % (folder_name, i), train_matrix, fmt='%1d')
        np.savetxt("%s/test_mtx-%d.txt" % (folder_name, i), test_matrix, fmt='%1d')
