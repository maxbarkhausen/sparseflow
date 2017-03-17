"""Logistic Regression estimator for binary classification."""
import tensorflow as tf
from ..functions import define_scope, lazy_property, calculate_penalty, convergence_stopping
from ..estimators.base_estimator import BaseEstimator


class LREstimator(BaseEstimator):
    def __init__(self, alpha=0.1, penalty='l2', learning_rate=0.1):
        super(LREstimator, self).__init__()
        self.alpha = alpha
        self.penalty = penalty
        self.learning_rate = learning_rate

    @property
    def model_params_dict(self):
        model_params_dict = {
            'num_features': self.num_features,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'penalty': self.penalty}
        return model_params_dict

    def _init_model(self):
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(None, self.num_features))
            self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.optimize

    @define_scope
    def theta(self):
        theta = tf.Variable(tf.truncated_normal(
            shape=(self.num_features, 1), stddev=0.01, dtype=tf.float32))
        return theta

    @define_scope
    def b(self):
        b = tf.Variable(tf.ones(shape=(1), dtype=tf.float32))
        return b

    @define_scope
    def predictions(self):
        predictions = tf.sigmoid(tf.matmul(self.X, self.theta) + self.b)
        return predictions

    @lazy_property
    def _penalty(self):
        return calculate_penalty(self, [self.theta])

    @define_scope
    def optimize(self):
        optimizer = tf.train.FtrlOptimizer(
            self.learning_rate).minimize(self.loss)
        return optimizer

    def stopping_criterion(self, train_loss, val_loss):
        message = convergence_stopping(self, train_loss)
        return message
