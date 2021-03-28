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
        node_shape, filter_shape = input_shape
        self.num_nodes = node_shape[1]
        self.num_features = node_shape[2]
        self.num_filters = filter_shape[1]

        #initialize all necessary weights and kernels
        self.reduction = self.add_weight(name = 'reduction',
                                         shape = filter_shape[1:],
                                         initializer = self.reduction_initializer,
                                         regularizer = self.reduction_regularizer,
                                         constraint = self.reduction_constraint,
                                         trainable = True)
        super(FairReductionGraphConv, self).build(input_shape)

    def call(self, inputs):
        nodes, filters = inputs
        #nodes has shape (batch, nodes, features)
        #filters has shape (batch, filters, nodes, nodes)

        #expand inputs along filter axis
        nodes = tf.expand_dims(nodes, axis = 1)      
        #nodes has shape (batch, 1, nodes, features)

        #apply fair sparsification
        fair_filters = tf.nn.relu(filters - tf.nn.sigmoid(self.reduction))
        #fair_filters has shape (batch, filters, nodes, nodes)

        #perform convolution
        conv_op = tf.matmul(fair_filters, nodes)
        conv_op = tf.reshape(tf.transpose(conv_op, (0, 2, 1, 3)), (-1, self.num_nodes, self.num_filters * self.num_features))
        #conv_op has shape (batch, nodes, filters * features)
        
        return conv_op, fair_filters