"""Factorization machine for binary classification."""
import tensorflow as tf
from ..functions import define_scope, lazy_property, calculate_penalty, \
    convergence_stopping, early_validation_stopping
from ..estimators.base_estimator import BaseEstimator


class FMEstimator(BaseEstimator):
    def __init__(self, num_dimensions=8, penalty='l2', alpha=0.00001,
                 learning_rate=0.01):
        super(FMEstimator, self).__init__()
        self.penalty = penalty
        self.alpha = alpha
        self.num_dimensions = num_dimensions
        self.present_loss = 0
        self.stopping_counter = 0
        self.learning_rate = learning_rate

    @property
    def model_params_dict(self):
        model_params_dict = {
            'num_dimensions': self.num_dimensions,
            'num_features': self.num_features,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'penalty': self.penalty}
        return model_params_dict

    def _init_model(self):
        try:
            self.load_model_parameters()
        except:
            pass
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(None, self.num_features))
            self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.optimize

    @define_scope
    def w_0(self):
        w_0 = tf.Variable(tf.ones(shape=(1), dtype=tf.float32))
        return w_0

    @define_scope
    def W(self):
        W = tf.Variable(tf.truncated_normal(
            shape=(self.num_features, 1), stddev=0.001, dtype=tf.float32))
        return W

    @define_scope
    def V(self):
        V = tf.Variable(tf.truncated_normal(
            shape=(self.num_features, self.num_dimensions), stddev=0.01), dtype=tf.float32)
        return V

    @define_scope
    def feature_weights(self):
        feature_weights = tf.matmul(self.X, self.W)
        return feature_weights

    @define_scope
    def factorization(self):
        term1 = (tf.square(tf.matmul(self.X, self.V)))
        term2 = tf.matmul(tf.square(self.X), tf.square(self.V))
        result = 0.5 * tf.reduce_sum(term1 - term2, axis=1, keep_dims=True)
        return result

    @define_scope
    def predictions(self):
        predictions = tf.sigmoid(self.feature_weights + self.factorization + self.w_0)
        return predictions

    @lazy_property
    def _penalty(self):
        penalty = calculate_penalty(self, [self.W, self.w_0, self.V])
        return penalty

    @define_scope
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)
        return optimizer

    def stopping_criterion(self, train_loss, val_loss):
        if val_loss is not None:
            message = early_validation_stopping(self, val_loss)
        else:
            message = convergence_stopping(self, train_loss)
        return message
