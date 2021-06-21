import tensorflow as tf

class FairCommunityAdditionGraphConv(tf.keras.layers.Layer):
    def __init__(self, num_additions,
                 addition_initializer=None,
                 addition_regularizer=None,
                 addition_constraint=None,
                 connection_initializer=None,
                 connection_regularizer=None,
                 connection_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FairCommunityAdditionGraphConv, self).__init__(**kwargs)

        self.num_additions = num_additions
        self.addition_initializer = tf.keras.initializers.get(addition_initializer)
        self.addition_regularizer = tf.keras.regularizers.get(addition_regularizer)
        self.addition_constraint = tf.keras.constraints.get(addition_constraint)
        self.connection_initializer = tf.keras.initializers.get(connection_initializer)
        self.connection_regularizer = tf.keras.regularizers.get(connection_regularizer)
        self.connection_constraint = tf.keras.constraints.get(connection_constraint)

    def build(self, input_shape):
        assert len(input_shape) == 2
        node_shape, adj_shape = input_shape
        self.num_nodes = node_shape[1]
        self.num_features = node_shape[2]

        #initialize all necessary weights and kernels
        self.addition = self.add_weight(name = 'addition',
                                        shape = (self.num_additions, self.num_features),
                                        initializer = self.addition_initializer,
                                        regularizer = self.addition_regularizer,
                                        constraint = self.addition_constraint,
                                        trainable = True)
        self.connection = self.add_weight(name = 'connection',
                                          shape = (self.num_nodes, self.num_additions),
                                          initializer = self.connection_initializer,
                                          regularizer = self.connection_regularizer,
                                          constraint = self.connection_constraint,
                                          trainable = True)
        super(FairCommunityAdditionGraphConv, self).build(input_shape)

    def call(self, inputs):
        nodes, adj = inputs
        #nodes has shape (batch, nodes, features)
        #adj has shape (batch, nodes, nodes)

        #perform convolution
        conv_op = tf.matmul(adj, nodes)
        #conv_op has shape (batch, nodes, features)

        #perform convolution for fairness additions
        fair_conv_op = tf.matmul(self.connection, self.addition)
        #fair_conv_op has shape (nodes, features)

        #apply fairness addition
        fair_op = conv_op + fair_conv_op
        #fair_op has shape (batch, nodes, features)

        return fair_op, adj