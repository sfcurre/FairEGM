import tensorflow as tf

class FairReductionGraphConv(tf.keras.layers.Layer):
    def __init__(self,
                 reduction_initializer=None,
                 reduction_regularizer=None,
                 reduction_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FairReductionGraphConv, self).__init__(**kwargs)

        self.reduction_initializer = tf.keras.initializers.get(reduction_initializer)
        self.reduction_regularizer = tf.keras.regularizers.get(reduction_regularizer)
        self.reduction_constraint = tf.keras.constraints.get(reduction_constraint)

    def build(self, input_shape):
        assert len(input_shape) == 2
        node_shape, adj_shape = input_shape
        self.num_nodes = node_shape[1]
        self.num_features = node_shape[2]

        #initialize all necessary weights and kernels
        self.reduction = self.add_weight(name = 'reduction',
                                         shape = (self.num_nodes, self.num_nodes),
                                         initializer = self.reduction_initializer,
                                         regularizer = self.reduction_regularizer,
                                         constraint = self.reduction_constraint,
                                         trainable = True)
        super(FairReductionGraphConv, self).build(input_shape)

    def call(self, inputs):
        nodes, adj = inputs
        #nodes has shape (batch, nodes, features)
        #adj has shape (batch, nodes, nodes)

        #apply fair sparsification
        fair_adj = tf.nn.relu(adj - tf.nn.sigmoid(self.reduction))
        #fair_adj has shape (batch, nodes, nodes)

        #perform convolution
        conv_op = tf.matmul(fair_adj, nodes)
        #conv_op has shape (batch, nodes, features)
        
        return conv_op, fair_adj