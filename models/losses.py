import tensorflow as tf

def dp_link_divergence_loss(attributes, edges):
    #edges is (batch, nodes, nodes)
    #atrributes is (batch, nodes, attributes)
    edges = tf.nn.sigmoid(edges)
    f = tf.matmul(edges, attributes) + 1e-7
    e = tf.reduce_sum(attributes, axis = 1) + 1e-7
    norme = tf.reduce_sum(e, axis = -1, keepdims=True)
    normf = tf.reduce_sum(f, axis = -1, keepdims=True)
    retval = tf.losses.kld(e / norme, f / normf)
    return retval

def build_reconstruction_loss(pos_weight):

    def reconstruction_loss(true_adj, pred_adj):
        true_adj = tf.cast(true_adj, tf.float32)
        return tf.nn.weighted_cross_entropy_with_logits(true_adj, pred_adj, pos_weight)
        
    return reconstruction_loss

def build_dp_link_divergence_loss_vgae(embedding_dim):

    def dp_loss(attributes, pred):
        #edges is (batch, nodes, nodes)
        #atrributes is (batch, nodes, attributes)
        edges = tf.nn.sigmoid(pred[...,:-2*embedding_dim])
        f = tf.matmul(edges, attributes) + 1e-7
        e = tf.reduce_sum(attributes, axis = 1) + 1e-7
        norme = tf.reduce_sum(e, axis = -1, keepdims=True)
        normf = tf.reduce_sum(f, axis = -1, keepdims=True)
        retval = tf.losses.kld(e / norme, f / normf)
        return retval

    return dp_loss

def build_reconstruction_loss_vgae(pos_weight, norm, num_nodes, embedding_dim):

    def reconstruction_loss(true_adj, pred):
        true_adj = tf.cast(true_adj, tf.float32)
        
        pred_adj = pred[..., :num_nodes]
        z_mean = pred[..., num_nodes:-embedding_dim]
        z_log_std = pred[..., -embedding_dim:]

        ent = norm * tf.nn.weighted_cross_entropy_with_logits(true_adj, pred_adj, pos_weight)
        kl = (0.5 / num_nodes) + tf.reduce_sum(1 + 2 * z_log_std - tf.square(z_mean) - tf.square(tf.exp(z_log_std)), 2)

        return ent - kl
        
    return reconstruction_loss

    