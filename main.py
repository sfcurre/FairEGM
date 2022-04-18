import numpy as np
import os, json
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict
from functools import singledispatch
import argparse, time

from layers.graph_cnn import GraphCNN
from layers.gfo_graph_conv import GFOGraphConv
from layers.cfo_graph_conv import CFOGraphConv
from layers.few_graph_conv import FEWGraphConv
from layers.link_prediction import LinkPrediction
from layers.link_reconstruction import LinkReconstruction
from models.fair_model import FairModel
from models.losses import *
from models.metrics import dp_link_divergence, recall_at_k, dp_at_k
from preprocess.read_data import read_data

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(5429)
tf.random.set_seed(5429)

TARGETED_FAIRNESS = lambda: GFOGraphConv()
COMMUNITY_FAIRNESS_10 = lambda: CFOGraphConv(10)
COMMUNITY_FAIRNESS_100 = lambda: CFOGraphConv(100)
SPARSE_FAIRNESS = lambda: FEWGraphConv(kernel_initializer=tf.keras.initializers.Ones())

@singledispatch
def to_serializable(val):
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    return np.float64(val)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', 
                        help='The dataset to train models on, one of [bail, cora, citeseer, facebook, pubmed].')
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
                        help='Number of folds for k-fold cross validation.')
    parser.add_argument('-Lp', '--lambda-param', type=float, default=1, 
                        help='The learning rate multiplier for the fair loss.')
    parser.add_argument('-Le', '--lambda-epochs', type=int, default=1, 
                        help='The number of epochs for the fair loss.')
    parser.add_argument('-i', '--init', type=str, default='glorot_normal',
                        help='Initialization to use for the fairness weights.')
    parser.add_argument('-i2', '--init2', type=str, default='glorot_normal',
                        help='Initialization to use for the fairness weight connections.')
    parser.add_argument('-r', '--random', type=int, default=0,
                        help='Number of random trials to run.')
    parser.add_argument('-c', '--constraint', type=str, default='',
                        help='Constraint method for fairness weights.')
    parser.add_argument('--vgae', action='store_true',
                        help='Whether or not to use the vgae architecture (default is gae).')
    return parser.parse_args()

def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(axis=1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

def reconstruction_model(num_nodes, num_features, embedding_dim):
    # num_features == embedding_dim
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((num_nodes, num_nodes))
    embeddings, conv_edges = GraphCNN(embedding_dim, activation='linear')([nodes, edges])
    output = LinkReconstruction()(embeddings)
    return tf.keras.models.Model([nodes, edges], [output, embeddings])

def reconstruction_model_variational(num_nodes, num_features, embedding_dim):
    # num_features == embedding_dim
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((num_nodes, num_nodes))
    z_mean, conv_edges = GraphCNN(embedding_dim, activation='linear')([nodes, edges])
    z_log_std, conv_edges = GraphCNN(embedding_dim, activation='linear')([nodes, edges])
    z = tf.keras.layers.Lambda(lambda x: x[0] + tf.random.normal([num_nodes, embedding_dim]) * tf.exp(x[1]))([z_mean, z_log_std])
    output = LinkReconstruction()(z)
    
    output = tf.keras.layers.Concatenate()([output, z_mean, z_log_std])

    model = tf.keras.models.Model([nodes, edges], [output, z])
    return model

def base_model(num_nodes, num_features, embedding_dim1, embedding_dim2):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(embedding_dim1, activation='relu')([nodes, edges])
    embeddings, conv_edges = GraphCNN(embedding_dim2, activation='linear')([conv_nodes, conv_edges])
    output = LinkReconstruction()(embeddings)
    return tf.keras.models.Model([nodes, edges], output), tf.keras.models.Model([nodes, edges], [embeddings, conv_edges])

def base_model_variational(num_nodes, num_features, embedding_dim1, embedding_dim2):
    nodes = tf.keras.layers.Input((num_nodes, num_features))
    edges = tf.keras.layers.Input((num_nodes, num_nodes))

    conv_nodes, conv_edges = GraphCNN(embedding_dim1, activation='relu')([nodes, edges])
    z_mean, conv_edges = GraphCNN(embedding_dim2, activation='linear')([nodes, edges])
    z_log_std, conv_edges = GraphCNN(embedding_dim2, activation='linear')([nodes, edges])
    z = tf.keras.layers.Lambda(lambda x: x[0] + tf.random.normal([num_nodes, embedding_dim2]) * tf.exp(x[1]))([z_mean, z_log_std])
    output = LinkReconstruction()(z)
    output = tf.keras.layers.Concatenate()([output, z_mean, z_log_std])

    return tf.keras.models.Model([nodes, edges], output), tf.keras.models.Model([nodes, edges], [z, conv_edges])

def kfold_base_model(all_features, fold_names, all_attributes, args, embedding_file='', augment_loss=False, return_weights=False):
    results = []
    for i, (train_edges, test_edges) in enumerate(fold_names):
        rdict = {}

        train_edges = np.load(train_edges)
        test_edges = np.load(test_edges)

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[None, use_node, :], train_edges[None, use_node][..., use_node]
        test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]
        norm_edges = preprocess_graph(train_edges[0])[None,...]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        
        if args.vgae:
            dp_loss = build_dp_link_divergence_loss_vgae(args.embedding_dim2)
            norm = (train_edges.shape[-1] * train_edges.shape[-1] / 
                    float((train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) * 2))
            def dp_metric(y_true, y_pred):
                return dp_loss(attributes.astype(np.float32), y_pred)
            base, base_embedding = base_model_variational(*features.shape[-2:], args.embedding_dim, args.embedding_dim2)
            base.compile(tf.keras.optimizers.Adam(args.learning_rate),
                         build_reconstruction_loss_vgae(pos_weight, norm, features.shape[1], args.embedding_dim2), [dp_metric])
        elif augment_loss:
            def dp_metric(y_true, y_pred):
                return dp_link_divergence_loss(attributes.astype(np.float32), y_pred)
            recon_metric = build_reconstruction_loss(pos_weight)
            def augmented_loss(y_true, y_pred):
                return recon_metric(y_true, y_pred) + augment_loss * dp_metric(y_true, y_pred)
            base, base_embedding = base_model(*features.shape[-2:], args.embedding_dim, args.embedding_dim2)
            base.compile(tf.keras.optimizers.Adam(args.learning_rate), augmented_loss, [recon_metric, dp_metric])
        else:
            def dp_metric(y_true, y_pred):
                return dp_link_divergence_loss(attributes.astype(np.float32), y_pred)
            base, base_embedding = base_model(*features.shape[-2:], args.embedding_dim, args.embedding_dim2)
            base.compile(tf.keras.optimizers.Adam(args.learning_rate), build_reconstruction_loss(pos_weight), [dp_metric])
            
        start_time = time.time()
        history = base.fit([features, norm_edges], train_edges, epochs = args.epochs, verbose = 0).history
        end_time = time.time()
        rdict['time'] = end_time - start_time
        history['task loss'] = history.pop('loss')
        history['fair loss'] = history.pop('dp_metric')
        rdict['history'] = history
        #not actually fair for base
        fair_nodes, _ = base_embedding.predict([features, norm_edges])
        fair_nodes = fair_nodes[0]
        if embedding_file:
            np.save(embedding_file + f'_{i}.npy', fair_nodes)
        if augment_loss:
            augmented_loss, recon_loss, dp_loss = base.evaluate([features, norm_edges], train_edges)
            rdict['augmented loss'] = augmented_loss
        else:    
            recon_loss, dp_loss = base.evaluate([features, norm_edges], train_edges)
        rdict['reconstruction loss'] = recon_loss
        rdict['link divergence'] = dp_loss
        for k in args.top_k:
            rdict[f'recall@{k}'] = recall_at_k(fair_nodes, test_edges, k=k)
            rdict[f'dp@{k}'] = dp_at_k(fair_nodes, attributes[0], k=k)
        if return_weights:
            rdict['fair_weights'] = None
            rdict['attributes'] = attributes[0]
            rdict['features'] = features[0]
            rdict['edges'] = train_edges[0]
            rdict['norm_edges'] = norm_edges[0]
            rdict['embedding'] = fair_nodes
            return rdict
        results.append(rdict)
    return results

def kfold_fair_model(all_features, fold_names, all_attributes, layer_constructor, args, embedding_file='', return_weights=False):
    results = []
    for i, (train_edges, test_edges) in enumerate(fold_names):
        rdict = {}
        
        train_edges = np.load(train_edges)
        test_edges = np.load(test_edges)

        use_node = np.any((train_edges != 0), axis = -1)

        features, train_edges = all_features[None, use_node, :], train_edges[None, use_node][..., use_node]
        test_edges, attributes = test_edges[use_node][..., use_node], all_attributes[None, use_node, :]
        norm_edges = preprocess_graph(train_edges[0])[None,...]

        pos_weight = float(train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) / train_edges.sum()
        
        if args.vgae:
            norm = (train_edges.shape[-1] * train_edges.shape[-1] / 
                    float((train_edges.shape[-1] * train_edges.shape[-1] - train_edges.sum()) * 2))
            task_loss = build_reconstruction_loss_vgae(pos_weight, norm, features.shape[1], args.embedding_dim2)
            fair_loss = build_dp_link_divergence_loss_vgae(args.embedding_dim2)
            
            model = FairModel(*features.shape[-2:], layer_constructor(), tf.keras.layers.Dense(args.embedding_dim, activation='relu'),
                               reconstruction_model_variational(features.shape[-2], args.embedding_dim, args.embedding_dim2))
        
            model.compile(tf.keras.optimizers.Adam(args.learning_rate), tf.keras.optimizers.Adam(args.learning_rate * args.lambda_param),
                          task_loss, fair_loss)
        else:
            model = FairModel(*features.shape[-2:], layer_constructor(), tf.keras.layers.Dense(args.embedding_dim, activation='relu'),
                               reconstruction_model(features.shape[-2], args.embedding_dim, args.embedding_dim2))
            model.compile(tf.keras.optimizers.Adam(args.learning_rate), tf.keras.optimizers.Adam(args.learning_rate * args.lambda_param),
                          build_reconstruction_loss(pos_weight), dp_link_divergence_loss)        
        
        start_time = time.time()
        history = model.fit(features, norm_edges, train_edges, attributes, args.epochs, args.lambda_epochs, verbose=0)
        end_time = time.time()
        rdict['time'] = end_time - start_time
        rdict['history'] = history
        fair_nodes, fair_edges = model.predict_embeddings([features, norm_edges])
        fair_nodes = fair_nodes[0]
        if embedding_file:
            np.save(embedding_file + f'_{i}.npy', fair_nodes)
        recon_loss, dp_loss = model.evaluate(features, norm_edges, train_edges, attributes)
        rdict['reconstruction loss'] = recon_loss
        rdict['link divergence'] = dp_loss
        for k in args.top_k:
            rdict[f'recall@{k}'] = recall_at_k(fair_nodes, test_edges, k=k)
            rdict[f'dp@{k}'] = dp_at_k(fair_nodes, attributes[0], k=k)
        print(f'Model {i+1}: [{rdict["reconstruction loss"]},{rdict["link divergence"]}]')
        if return_weights:
            rdict['fair_weights'] = model.fair_layer.get_weights()
            rdict['attributes'] = attributes[0]
            rdict['features'] = features[0]
            rdict['edges'] = train_edges[0]
            rdict['norm_edges'] = norm_edges[0]
            rdict['embedding'] = fair_nodes
            return rdict
        results.append(rdict)
        
    return results

def main():
    args = parse_args()
    if args.embedding_dim2 == 0:
        args.embedding_dim2 = args.embedding_dim

    features, edge_gen, attributes = read_data(args.dataset, args.folds, args.random)
    
    fold_names = []
    for i, (train, test) in enumerate(edge_gen):
        np.save(f'./data/{args.dataset}/folds{("_r" if args.random else "")}/fold{i}_train.npy', train)
        np.save(f'./data/{args.dataset}/folds{("_r" if args.random else "")}/fold{i}_test.npy', test)
        fold_names.append((f'./data/{args.dataset}/folds{("_r" if args.random else "")}/fold{i}_train.npy',
                           f'./data/{args.dataset}/folds{("_r" if args.random else "")}/fold{i}_test.npy'))
    
    addon = f'd-{args.embedding_dim}_d2-{args.embedding_dim2}_i-{args.init}_i2-{args.init2}'

    con = None
    if args.constraint:
        addon += f'_c-{args.constraint}'
        con = getattr(tf.keras.constraints, args.constraint)()
        # if type(con) is tf.keras.constraints.Constraint:
        #     con = con()

    if args.lambda_epochs != 1:
        addon += f'_Le-{args.lambda_epochs}'

    if args.vgae:
        addon += f'_m-VGAE'

    TARGETED_FAIRNESS = lambda: GFOGraphConv(addition_initializer=args.init, addition_constraint=con)
    COMMUNITY_FAIRNESS_10 = lambda: CFOGraphConv(10, addition_initializer=args.init, addition_constraint=con,
                                                 connection_initializer=args.init2, connection_constraint=con)
    COMMUNITY_FAIRNESS_100 = lambda: CFOGraphConv(100, addition_initializer=args.init, addition_constraint=con, 
                                                  connection_initializer=args.init2, connection_constraint=con)
    SPARSE_FAIRNESS = lambda: FEWGraphConv(kernel_initializer=tf.keras.initializers.Ones(), kernel_constraint=con)

    results = {}
    results['base'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/base_{addon}')
    results['gfo'] = kfold_fair_model(features, fold_names, attributes, TARGETED_FAIRNESS, args, embedding_file=f'./results/{args.dataset}/embeddings/gfo_{addon}')
    results['cfo_10'] = kfold_fair_model(features, fold_names, attributes, COMMUNITY_FAIRNESS_10, args, embedding_file=f'./results/{args.dataset}/embeddings/cfo10_{addon}')
    results['cfo_100'] = kfold_fair_model(features, fold_names, attributes, COMMUNITY_FAIRNESS_100, args, embedding_file=f'./results/{args.dataset}/embeddings/cfo100_{addon}')
    results['few'] = kfold_fair_model(features, fold_names, attributes, SPARSE_FAIRNESS, args, embedding_file=f'./results/{args.dataset}/embeddings/few_{addon}')
    results['augmented'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/augmented_{addon}', augment_loss=True)
    results['augmented10'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/augmented10_{addon}', augment_loss=10)
    results['augmented100'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/augmented100_{addon}', augment_loss=100)
    results['augmented1000'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/augmented1000_{addon}', augment_loss=1000)
    results['augmented10000'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/augmented10000_{addon}', augment_loss=10000)
    results['augmented1000'] = kfold_base_model(features, fold_names, attributes, args, embedding_file=f'./results/{args.dataset}/embeddings/augmented100000_{addon}', augment_loss=100000)

    with open(f'./results/{args.dataset}/results_{addon}.json', 'w') as fp:
        json.dump(results, fp, indent=True, default=to_serializable)


if __name__ == '__main__':
    main()