"Neural network estimator for binary classification."
import tensorflow as tf
from ..estimators.base_estimator import BaseEstimator
from ..functions import define_scope, convergence_stopping, early_validation_stopping
from ..model_functions import xavier_weights, biases


class ENNEstimator(BaseEstimator):
    def __init__(self, embedding_size=256, lin_size=2048,
                 hidden1_size=1024, hidden2_size=1024, output_size=512, learning_rate=0.01,
                 feature_indices=None):
        super(ENNEstimator, self).__init__()
        self.embedding_size = embedding_size
        self.lin_size = lin_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.stopping_counter = 0
        self.learning_rate = learning_rate
        self.feature_indices = feature_indices

    @property
    def model_params_dict(self):
        model_params_dict = {
            'embedding_size': self.embedding_size,
            'lin_size': self.lin_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'output_size': self.output_size,
            'feature_indices': self.feature_indices
        }
        return model_params_dict

    @property
    def num_fields(self):
        return self.feature_indices.shape[0] - 1

    def _init_model(self):
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(None, self.num_features))
            self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.optimize

    @define_scope
    def embedding_weights(self):
        embedding_weights = []
        for i in range(0, self.num_fields):
            field_length = self.feature_indices[i + 1] - self.feature_indices[i]
            field_weights = xavier_weights((field_length, self.embedding_size))
            embedding_weights.append(field_weights)
        return embedding_weights

    @define_scope
    def lin_weights(self):
        return xavier_weights((self.embedding_size * self.num_fields, self.lin_size))

    @define_scope
    def lin_biases(self):
        return biases((self.lin_size,))

    @define_scope
    def hidden1_weights(self):
        return xavier_weights((self.lin_size, self.hidden1_size))

    @define_scope
    def hidden1_biases(self):
        return biases((self.hidden1_size,))

    @define_scope
    def hidden2_weights(self):
        return xavier_weights((self.hidden1_size, self.hidden2_size))

    @define_scope
    def hidden2_biases(self):
        return biases((self.hidden2_size,))

    @define_scope
    def output_weights(self):
        return xavier_weights((self.hidden2_size, 1))

    @define_scope
    def output_bias(self):
        return biases((1,))

    @define_scope
    def embeddings(self):
        embeddings = []
        for i in range(0, self.num_fields):
            field_beginning = self.feature_indices[i]
            field_end = self.feature_indices[i + 1]
            field_slice = tf.slice(self.X, [0, field_beginning], [-1, field_end - field_beginning])
            embedding = tf.matmul(field_slice, self.embedding_weights[i])
            embeddings.append(embedding)
        embeddings = tf.stack(embeddings, axis=1)
        return embeddings

    @define_scope
    def lin_layer(self):
        unrolled_embeddings = tf.reshape(self.embeddings, [-1, self.num_fields * self.embedding_size])
        lin_part = tf.matmul(unrolled_embeddings, self.lin_weights)
        lin_layer = self.lin_biases + lin_part
        return lin_layer

    @define_scope
    def hidden1(self):
        hidden1 = tf.nn.relu(tf.matmul(self.lin_layer, self.hidden1_weights)) + self.hidden1_biases
        hidden1 = tf.nn.dropout(hidden1, keep_prob=0.7)
        return hidden1

    @define_scope
    def hidden2(self):
        hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.hidden2_weights)) + self.hidden2_biases
        hidden2 = tf.nn.dropout(hidden2, keep_prob=0.7)
        return hidden2

    @define_scope
    def output(self):
        output = tf.matmul(self.hidden2, self.output_weights) + self.output_bias
        output = tf.nn.dropout(output, keep_prob=0.7)
        return output

    @define_scope
    def predictions(self):
        predictions = tf.sigmoid(self.output)
        return predictions

    @define_scope
    def optimize(self):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.96, staircase=True)
        _optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum=0.9).minimize(self.loss)
        return _optimizer

    def stopping_criterion(self, train_loss, val_loss):
        if val_loss is not None:
            message = early_validation_stopping(self, val_loss)
        else:
            message = convergence_stopping(self, train_loss)
        return message
