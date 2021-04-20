import tensorflow as tf

class LinkPrediction(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LinkPrediction, self).__init__(**kwargs)

        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def get_config(self):
        config = super(LinkPrediction, self).get_config()        
        config.update({'units': self.units,
                       'activation': self.activation,
                       'use_bias': self.use_bias})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        # (batch, nodes, features)
        assert len(input_shape) == 3
        self.node1_kernel = self.add_weight(name = 'node1_kernel',
                                            shape = (input_shape[-1], self.units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.node2_kernel = self.add_weight(name = 'node2_kernel',
                                            shape = (input_shape[-1], self.units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.node1_bias = self.add_weight(name = 'node1_bias',
                                            shape = (self.units,),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        self.node2_bias = self.add_weight(name = 'node2_bias',
                                            shape = (self.units,),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)

        super(LinkPrediction, self).build(input_shape)

    def call(self, inputs):
        node1 = tf.matmul(inputs, self.node1_kernel) + (self.node1_bias if self.use_bias else 0)
        node2 = tf.matmul(inputs, self.node2_kernel) + (self.node2_bias if self.use_bias else 0)
        output = tf.matmul(node1, tf.transpose(node2, (0, 2, 1)))
        return self.activation(output)