import tensorflow as tf

class FairModel:
    def __init__(self, num_nodes, num_features, fair_layer, dense_layer, task_model):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.dense_layer = dense_layer
        self.fair_layer = fair_layer
        self.task_model = task_model
        self.model = self.build()
        self.compiled = False

    def build(self):
        nodes = tf.keras.layers.Input((self.num_nodes, self.num_features))
        edges = tf.keras.layers.Input((1, self.num_nodes, self.num_nodes))

        fair_conv, fair_edges = self.fair_layer([nodes, edges])
        fair_nodes = tf.keras.layers.TimeDistributed(self.dense_layer)(fair_conv)

        output = self.task_model([fair_nodes, fair_edges])
        return tf.keras.models.Model([nodes, edges], output)

    def compile(self, optimizer, task_loss, fair_loss, metrics):
        self.optimizer = optimizer
        self.task_loss = task_loss
        self.fair_loss = fair_loss
        self.metrics = metrics
        self.compiled = True

    def fit(self, nodes, edges, target, sensitive_attributes, epochs):
        assert self.compiled, "Model must be compiled before use"
        for i in range(epochs):
            print(f'Epoch {i+1}/{epochs}:')
            fl = tl = 0

            #fit fairness
            self.model.trainable = False
            self.fair_layer.trainable = True
            self.model.compile(self.optimizer, self.fair_loss, self.metrics)

            fl = self.model.train_on_batch([nodes, edges], sensitive_attributes)

            #fit task
            self.model.trainable = True
            self.fair_layer.trainable = False
            self.model.compile(self.optimizer, self.task_loss, self.metrics)

            tl = self.model.train_on_batch([nodes, edges], target)

            print(f'Fairness - {fl}')
            print(f'Task     - {tl}')

    def get_fair_modifications(self):
        return self.fair_layer.get_weights()