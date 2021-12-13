import tensorflow as tf
import gc

class FairModel:
    def __init__(self, num_nodes, num_features, fair_layer, dense_layer, task_model):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.dense_layer = dense_layer
        self.fair_layer = fair_layer
        self.task_model = task_model
        self.model, self.embedding_model = self.build()

        self.compiled = False

    def build(self):
        nodes = tf.keras.layers.Input((self.num_nodes, self.num_features))
        edges = tf.keras.layers.Input((self.num_nodes, self.num_nodes))

        fair_conv, fair_edges = self.fair_layer([nodes, edges])
        fair_nodes = tf.keras.layers.TimeDistributed(self.dense_layer)(fair_conv)

        outputs = self.task_model([fair_nodes, fair_edges])
        return tf.keras.models.Model([nodes, edges], outputs[:-1]), tf.keras.models.Model([nodes, edges], [outputs[-1], fair_edges])

    def compile(self, task_optimizer, fair_optimizer, task_loss, fair_loss):
        self.task_optimizer = task_optimizer
        self.fair_optimizer = fair_optimizer
        self.task_loss = task_loss
        self.fair_loss = fair_loss
        self.compiled = True

    @tf.function
    def train_step(self, nodes, edges, target, sensitive_attributes, lambda_epochs):

        with tf.GradientTape() as task_tape:

            output = self.model([nodes, edges], training = True)
            task_loss = self.task_loss(target, output)
            
            self.model.trainable = True
            self.fair_layer.trainable = False

            task_gradients = task_tape.gradient(task_loss, self.model.trainable_variables)
            self.task_optimizer.apply_gradients(zip(task_gradients, self.model.trainable_variables))

        for i in range(lambda_epochs):
            with tf.GradientTape() as fair_tape: 

                output = self.model([nodes, edges], training = True)
                fair_loss = self.fair_loss(sensitive_attributes, output)
                
                self.model.trainable = False
                self.fair_layer.trainable = True

                fair_gradients = fair_tape.gradient(fair_loss, self.fair_layer.trainable_variables)
                self.fair_optimizer.apply_gradients(zip(fair_gradients, self.fair_layer.trainable_variables))

        tl = tf.reduce_mean(task_loss, axis = None)
        fl = tf.reduce_mean(fair_loss, axis = None)
        
        return tl, fl

    def evaluate(self, nodes, edges, target, sensitive_attributes):
        output = self.model([nodes, edges], training = False)

        task_loss = self.task_loss(target, output)
        fair_loss = self.fair_loss(sensitive_attributes, output)
        
        tl = tf.reduce_mean(task_loss, axis = None)
        fl = tf.reduce_mean(fair_loss, axis = None)
        
        return tl.numpy(), fl.numpy()

    def fit(self, nodes, edges, target, sensitive_attributes, epochs, lambda_epochs = 1, verbose = 1):
        assert self.compiled, "Model must be compiled before use"

        nodes = tf.constant(nodes, dtype = tf.float32)
        edges = tf.constant(edges, dtype = tf.float32)
        target = tf.constant(target, dtype = tf.float32)
        sensitive_attributes = tf.constant(sensitive_attributes, dtype = tf.float32)

        print_con = lambda x: ((x+1) % (epochs // 20)) == 0

        history = {'fair loss': [], 'task loss': []}

        for i in range(epochs):
            if verbose and print_con(i):
                print(f'Epoch {i+1}/{epochs}:')
            
            tl, fl = self.train_step(nodes, edges, target, sensitive_attributes, lambda_epochs)
            
            tl = tl.numpy()
            fl = fl.numpy()
            
            if verbose and print_con(i):
                print(f'Task     - {tl}')
                print(f'Fairness - {fl}')
                
            tf.keras.backend.clear_session()
            gc.collect()

            history['task loss'].append(tl)
            history['fair loss'].append(fl)
            
        return history

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_embeddings(self, *args, **kwargs):
        return self.embedding_model.predict(*args, **kwargs)

    def get_fair_modifications(self):
        return self.fair_layer.get_weights()