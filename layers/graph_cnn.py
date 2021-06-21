import tensorflow as tf

class GraphCNN(tf.keras.layers.Layer):
    def __init__(self, output_dim,
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
        super(GraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) == 2
        node_shape, adj_shape = input_shape
        self.num_nodes = node_shape[1]
        self.num_features = node_shape[2]

        #initialize all necessary weights and kernels
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (self.num_features, self.output_dim),
                                      initializer = self.kernel_initializer,
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint,
                                      trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                    shape = (self.output_dim,),
                                    initializer = self.bias_initializer,
                                    regularizer = self.bias_regularizer,
                                    constraint = self.bias_constraint,
                                    trainable = True)
        super(GraphCNN, self).build(input_shape)

    def call(self, inputs):
        nodes, adj = inputs
        #nodes has shape (batch, nodes, features)
        #adj has shape (batch, nodes, nodes)

        #perform convolution
        conv_op = tf.matmul(adj, nodes)
        #conv_op has shape (batch, nodes, features)

        conv_out = tf.matmul(conv_op, self.kernel)
        #conv_out is shape (batch, nodes, output_dim)

        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.bias)

        return self.activation(conv_out), adj

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        node_shape, _ = input_shape
        return node_shape[:2] + (self.output_dim,)
