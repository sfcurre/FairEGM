import tensorflow as tf

class FEWGraphConv(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FEWGraphConv, self).__init__(**kwargs)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) == 2
        node_shape, adj_shape = input_shape
        self.num_nodes = node_shape[1]
        self.num_features = node_shape[2]

        #initialize all necessary weights and kernels
        self.kernel = self.add_weight(name = 'kernels',
                                       shape = (self.num_nodes, self.num_nodes),
                                       initializer = self.kernel_initializer,
                                       regularizer = self.kernel_regularizer,
                                       constraint = self.kernel_constraint,
                                       trainable = True)
        super(FEWGraphConv, self).build(input_shape)

    def call(self, inputs):
        nodes, adj = inputs
        #nodes has shape (batch, nodes, features)
        #adj has shape (batch, nodes, nodes)

        #apply fair sparsification
        fair_adj = adj * self.kernel
        #fair_adj has shape (batch, nodes, nodes)

        #perform convolution
        conv_op = tf.matmul(fair_adj, nodes)
        #conv_op has shape (batch, nodes, features)
        
        return conv_op, fair_adj