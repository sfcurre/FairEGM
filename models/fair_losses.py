import tensorflow as tf

def dp_link_divergence_loss(attributes, edges):
    #edgess is (batch, filters, nodes, nodes)
    #atrributes is (batch, filters, nodes, attributes)
    f = tf.matmul(edges, attributes)
    #partial is (batch, filters, nodes)
    f = f / tf.reduce_sum(attributes, axis = 2, keepdims=True)
    #f is (batch, filters, attributes)
    e = tf.ones_like(f)
    retval = tf.losses.kld(e / tf.norm(e, axis = -1, ord = 1, keepdims=True),
                           f / tf.norm(f, axis = -1, ord = 1, keepdims=True))
    return retval