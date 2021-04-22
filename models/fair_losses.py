import tensorflow as tf

def dp_link_divergence_loss(attributes, edges):
    #edgess is (batch, nodes, nodes)
    #atrributes is (batch, nodes, attributes)
    f = tf.matmul(edges, attributes)
    #partial is (batch, nodes)
    f = f / tf.reduce_sum(attributes, axis = 1, keepdims=True)
    #f is (batch, attributes)
    e = tf.ones_like(f)
    f = tf.clip_by_value(f, 1e-7, 1)
    norme = tf.norm(e, axis = -1, ord = 1, keepdims=True)
    normf = tf.norm(f, axis = -1, ord = 1, keepdims=True)
    retval = tf.losses.kld(e / norme, f / normf)
    return retval