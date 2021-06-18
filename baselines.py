from __future__ import division
from __future__ import print_function

import time
import os, json

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import squareform, pdist
from functools import singledispatch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import sklearn.preprocessing as skpp

from baselines.gae.gae.optimizer import OptimizerAE, OptimizerVAE
from baselines.gae.gae.model import GCNModelAE, GCNModelVAE
from baselines.gae.gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from baselines.inform.method.debias_graph import DebiasGraph
from baselines.inform.utils import *
from baselines.FairWalk.main import fairwalk

import networkx as nx
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

from models.metrics import *
from preprocess.read_data import *

@singledispatch
def to_serializable(val):
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    return np.float64(val)

class LINE:
    def __init__(self, dimension=128, ratio=3200, negative=5, batch_size=1000, init_lr=0.025, seed=None):
        self.dimension = dimension
        self.ratio = ratio
        self.negative = negative
        self.batch_size = batch_size
        self.init_lr = init_lr
        if seed is not None:
            np.random.seed(seed)

    def train(self, graph):
        self.graph = graph
        self.is_directed = nx.is_directed(self.graph)
        self.num_node = graph.number_of_nodes()
        self.num_sampling_edge = self.ratio * self.num_node

        node2id = dict([(node, vid) for vid, node in enumerate(graph.nodes())])
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.graph.edges()]
        self.edges_prob = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()])
        self.edges_prob /= np.sum(self.edges_prob)
        self.edges_table, self.edges_prob = alias_setup(self.edges_prob)

        degree_weight = np.asarray([0] * self.num_node)
        for u, v in graph.edges():
            degree_weight[node2id[u]] += graph[u][v].get("weight", 1.0)
            if not self.is_directed:
                degree_weight[node2id[v]] += graph[u][v].get("weight", 1.0)
        self.node_prob = np.power(degree_weight, 0.75)
        self.node_prob /= np.sum(self.node_prob)
        self.node_table, self.node_prob = alias_setup(self.node_prob)

        self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
        self.fair_emb_vertex = self.emb_vertex.copy()
        self._train_line()
        self.embeddings = skpp.normalize(self.emb_vertex, "l2")
        return self.embeddings

    def _update(self, vec_u, vec_v, vec_error, label):
        # update vetex embedding and vec_error
        f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
        g = (self.lr * (label - f)).reshape((len(label), 1))
        vec_error += g * vec_v
        vec_v += g * vec_u

    def _train_line(self):
        self.lr = self.init_lr
        batch_size = self.batch_size
        num_batch = int(self.num_sampling_edge / batch_size)
        epoch_iter = range(num_batch)
        for b in epoch_iter:
            # if b % self.batch_size == 0:
            #     self.lr = self.init_lr * max((1 - b * 1.0 / num_batch), 0.0001)
            self.lr = self.init_lr * max((1 - b * 1.0 / num_batch), 0.0001)
            u, v = [0] * batch_size, [0] * batch_size
            for i in range(batch_size):
                edge_id = alias_draw(self.edges_table, self.edges_prob)
                u[i], v[i] = self.edges[edge_id]
                if not self.is_directed and np.random.rand() > 0.5:
                    v[i], u[i] = self.edges[edge_id]

            vec_error = np.zeros((batch_size, self.dimension))
            label, target = np.asarray([1 for i in range(batch_size)]), np.asarray(v)
            for j in range(self.negative + 1):
                if j != 0:
                    label = np.asarray([0 for i in range(batch_size)])
                    for i in range(batch_size):
                        target[i] = alias_draw(self.node_table, self.node_prob)
                self._update(
                    self.emb_vertex[u], self.emb_vertex[target], vec_error, label
                )
            self.emb_vertex[u] += vec_error

def run_gae(features, train_adj, test_adj, attributes, top_k, model_str):

    # Settings
    flags = type('FLAGS', (object,), {})
    flags.learning_rate = 0.01
    flags.epochs = 1000
    flags.hidden1 = 128
    flags.hidden2 = 128
    flags.weight_decay = 0
    flags.dropout = 0
    flags.features = 1
    FLAGS = flags

    # Store original adjacency matrix (without diagonal entries) for later
    # adj_orig = train_adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()
    adj = train_adj

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(sp.coo_matrix(features))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            pos_weight=pos_weight,
                            norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_scores(train_edges, test_edges, attributes, emb=None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = sigmoid(np.dot(emb, emb.T))
        
        rdict = {}
        rdict['reconstruction loss'] = build_reconstruction_metric(pos_weight)(train_edges, adj_rec)
        rdict['link divergence'] = dp_link_divergence(attributes[None, ...], adj_rec)
        rdict['recall@k'] = recall_at_k(emb, test_edges, k=top_k)
        rdict['dp@k'] = dp_at_k(emb, attributes, k=top_k)
        print(f'Model: [{rdict["reconstruction loss"]},{rdict["link divergence"]},{rdict["recall@k"]},{rdict["dp@k"]}]')
        
        return rdict

    adj_label = train_adj
    adj_label = sparse_to_tuple(sp.coo_matrix(adj_label))

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
            "train_acc=", "{:.5f}".format(avg_accuracy),
            "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    results = get_scores(train_adj, test_adj, attributes)
    return results

def kmeans_gae(all_features, train_folds, test_folds, all_attributes, top_k, model_str):
    results = []
    for i in range(len(train_folds)):
        train_edges = train_folds[i]
        test_edges = test_folds[i]

        use_node = np.any((train_edges != 0), axis = -1)
        features = all_features[use_node]
        attributes = all_attributes[use_node]
        train_edges = train_edges[use_node][:, use_node]
        test_edges = test_edges[use_node][:, use_node]

        results.append(run_gae(features, train_edges, test_edges, attributes, top_k, model_str))
    return results

def kmeans_inform(all_features, train_folds, test_folds, all_attributes, top_k, alpha = 0.5):
    results = []
    for i in range(len(train_folds)):
        rdict = {}
        train_edges = train_folds[i]
        test_edges = test_folds[i]

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[use_node], train_edges[use_node][:, use_node]
        test_edges, attributes = test_edges[use_node][:, use_node], all_attributes[use_node]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        sims = 1 - squareform(pdist(features, metric='cosine'))

        FairGraph = DebiasGraph()
        adj = FairGraph.line(sp.csc_matrix(train_edges),sp.csc_matrix(sims), alpha=alpha, maxiter=100, lr=0.001, tol=1e-6)
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph(), edge_attribute='weight')
        model = LINE(ratio=3200, seed=0)
        embs = model.train(graph)

        adj_rec = sims = 1 - squareform(pdist(embs, metric='cosine'))
        print(adj_rec)
        rdict = {}
        rdict['reconstruction loss'] = build_reconstruction_metric(pos_weight)(train_edges, adj_rec)
        rdict['link divergence'] = dp_link_divergence(attributes[None,...], adj_rec)
        rdict['recall@k'] = recall_at_k(embs, test_edges, k=top_k)
        rdict['dp@k'] = dp_at_k(embs, attributes, k=top_k)
        print(f'Model: [{rdict["reconstruction loss"]},{rdict["link divergence"]},{rdict["recall@k"]},{rdict["dp@k"]}]')
        
        results.append(rdict)
    
    return results

def kmeans_fairwalk(all_features, train_folds, test_folds, all_attributes, top_k, attr_id=0):
    results = []
    for i in range(len(train_folds)):
        train_edges = train_folds[i]
        test_edges = test_folds[i]

        use_node = np.any((train_edges != 0), axis=-1)
        print(use_node.sum())
        # features = all_features[use_node]
        attributes = all_attributes[use_node]
        train_edges = train_edges[use_node][:, use_node]
        test_edges = test_edges[use_node][:, use_node]
        pos_weight = float(train_edges.shape[0] * train_edges.shape[0] - train_edges.sum()) / train_edges.sum()

        embeddings = fairwalk(train_edges, test_edges, attributes, fold_id=i, attr_id=attr_id)

        # Evaluate the embeddings
        def get_scores(train_edges, test_edges, attributes, emb):
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Predict on test set of edges
            adj_rec = sigmoid(np.dot(emb, emb.T))

            rdict = {}
            rdict['reconstruction loss'] = build_reconstruction_metric(pos_weight)(train_edges, adj_rec)
            rdict['link divergence'] = dp_link_divergence(attributes[None, ...], adj_rec)
            rdict['recall@k'] = recall_at_k(emb, test_edges, k=top_k)
            rdict['dp@k'] = dp_at_k(emb, attributes, k=top_k)
            print(
                f'Model: [{rdict["reconstruction loss"]},{rdict["link divergence"]},{rdict["recall@k"]},{rdict["dp@k"]}]')

            return rdict
        results.append(get_scores(train_edges, test_edges, attributes, embeddings))
    return results

def main():
    folds = 5
    top_k = 20
    for dataset in ['cora', 'citeseer', 'facebook', 'pubmed']:
        if dataset == 'citeseer':
            data = read_citeseer(folds)
        elif dataset == 'cora':
            data = read_cora(folds)
        elif dataset == 'credit':
            data = read_credit(folds)
        elif dataset == 'facebook':
            data = read_facebook(folds)
        elif dataset == 'pubmed':
            data = read_pubmed(folds)

        results = {}
        for model in ['gae', 'vgae', 'inform']: # + ['graphsage']:
            if model == 'gae':
                rlist = kmeans_gae(*data, top_k, 'gcn_ae')
            elif model == 'vgae':
                rlist = kmeans_gae(*data, top_k, 'gcn_vae')
            elif model == 'inform':
                rlist = kmeans_inform(*data, top_k, alpha=0.5)
            elif model == 'fairwalk':
                rlist = kmeans_fairwalk(*data, top_k)

            results[model] = rlist

        # with open(f'./results/{dataset}/baseline_results.json', 'w') as fp:
        #     json.dump(results, fp, indent=True, default=to_serializable)

if __name__ == '__main__':
    main()