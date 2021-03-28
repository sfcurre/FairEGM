import tensorflow as tf

def fair_link_loss(attributes, links):
    #links is (batch, filters, nodes, nodes)
    #atrributes is (batch, filters, nodes, attributes)
    partial = tf.reduce_sum(links, axis = -1)
    #partial is (batch, filters, nodes)
    f = tf.squeeze(tf.matmul(tf.transpose(attributes, (0, 1, 3, 2)), partial[..., None]))
    #f is (batch, filters, attributes)
    e = tf.ones_like(f)
    return tf.losses.kld(f / tf.norm(f), e / tf.norm(e))
