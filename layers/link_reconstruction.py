import tensorflow as tf

class LinkReconstruction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LinkReconstruction, self).__init__(**kwargs)

    def build(self, input_shape):
        # (batch, nodes, features)
        assert len(input_shape) == 3
        super(LinkReconstruction, self).build(input_shape)

    def call(self, inputs):
        dots = tf.matmul(inputs, tf.transpose(inputs, (0, 2, 1)))
        dots = dots / tf.norm(inputs, axis = 1, keepdims = True)
        dots = dots / tf.norm(inputs, axis = 2, keepdims = True)
        return dots