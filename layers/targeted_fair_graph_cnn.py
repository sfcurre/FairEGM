import tensorflow as tf

class FairTargetedAdditionGraphConv(tf.keras.layers.Layer):
    def __init__(self,
                 addition_initializer=None,
                 addition_regularizer=None,
                 addition_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FairTargetedAdditionGraphConv, self).__init__(**kwargs)

        self.addition_initializer = tf.keras.initializers.get(addition_initializer)
        self.addition_regularizer = tf.keras.regularizers.get(addition_regularizer)
        self.addition_constraint = tf.keras.constraints.get(addition_constraint)

    def build(self, input_shape):
        assert len(input_shape) == 2
        node_shape, filter_shape = input_shape
        self.num_nodes = node_shape[1]
        self.num_features = node_shape[2]
        self.num_filters = filter_shape[1]

        #initialize all necessary weights and kernels
        self.addition = self.add_weight(name = 'addition',
                                        shape = (self.num_nodes, self.num_filters * self.num_features),
                                        initializer = self.addition_initializer,
                                        regularizer = self.addition_regularizer,
                                        constraint = self.addition_constraint,
                                        trainable = True)
        super(FairTargetedAdditionGraphConv, self).build(input_shape)

    def call(self, inputs):
        nodes, filters = inputs
        #nodes has shape (batch, nodes, features)
        #filters has shape (batch, filters, nodes, nodes)

        #expand inputs along filter axis
        nodes = tf.expand_dims(nodes, axis = 1)     
        #nodes has shape (batch, 1, nodes, features)

        #perform convolution
        conv_op = tf.matmul(filters, nodes)
        conv_op = tf.reshape(tf.transpose(conv_op, (0, 2, 1, 3)), (-1, self.num_nodes, self.num_filters * self.num_features))
        #conv_op has shape (batch, nodes, filters * features)

        #apply fairness addition
        fair_op = conv_op + self.addition
        #fair_op has shape (batch, nodes, filters * features)

        return fair_op, filters