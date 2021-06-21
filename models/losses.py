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

def dp_link_entropy_loss(attributes, edges):
    #edges is (batch, nodes, nodes)
    #atrributes is (batch, nodes, attributes)
    f = tf.matmul(edges, attributes)
    #partial is (batch, nodes)
    f = f / tf.reduce_sum(attributes, axis = 1, keepdims=True)
    #f is (batch, attributes)
    f = tf.clip_by_value(f, 1e-7, 1)
    normf = tf.reduce_sum(f, axis = -1, keepdims=True)
    f = (f / normf)
    return tf.reduce_mean(tf.reduce_sum(f * tf.math.log(f), axis = -1))

def build_reconstruction_loss(pos_weight):

    def reconstruction_loss(true_adj, pred_adj):
        true_adj = tf.cast(true_adj, tf.float32)
        return tf.nn.weighted_cross_entropy_with_logits(true_adj, pred_adj, pos_weight)
        
        # b_ce = tf.keras.losses.binary_crossentropy(true_adj, pred_adj)
        # weight_vector = true_adj * pos_weight + (1 - true_adj)
        # return tf.reduce_mean(weight_vector * b_ce)

    return reconstruction_loss