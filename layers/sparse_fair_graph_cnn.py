import tensorflow as tf

class FairReductionGraphCNN(tf.keras.layers.Layer):
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
                 reduction_initializer=None,
                 reduction_regularizer=None,
                 reduction_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FairReductionGraphCNN, self).__init__(**kwargs)

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
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (self.num_filters * self.num_features, self.output_dim),
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
        self.reduction = self.add_weight(name = 'reduction',
                                         shape = filter_shape,
                                         initializer = self.reduction_initializer,
                                         regularizer = self.reduction_regularizer,
                                         constraint = self.reduction_constraint,
                                         trainable = True)
        super(FairReductionGraphCNN, self).build(input_shape)

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

        conv_out = tf.matmul(conv_op, self.kernel)
        #conv_out is shape (batch, nodes, output_dim)

        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.bias)

        return self.activation(conv_out)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        node_shape, _ = input_shape
        return node_shape[:2] + (self.output_dim,)
