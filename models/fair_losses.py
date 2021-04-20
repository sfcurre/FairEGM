import tensorflow as tf

def dp_link_divergence_loss(attributes, edges):
    #edgess is (batch, nodes, nodes)
    #atrributes is (batch, nodes, attributes)
    f = tf.matmul(edges, attributes)
    #partial is (batch, nodes)
    f = f / tf.reduce_sum(attributes, axis = 1, keepdims=True)
    #f is (batch, attributes)
    e = tf.ones_like(f)
    retval = tf.losses.kld(e / tf.norm(e, axis = -1, ord = 1, keepdims=True),
                           f / tf.norm(f, axis = -1, ord = 1, keepdims=True))
    return retval