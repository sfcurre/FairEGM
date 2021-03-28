import tensorflow as tf

def fair_link_loss(attributes, links):
    return (tf.matmul(attributes, tf.transpose(attributes, (0, 1, 3, 2))) - links) ** 2