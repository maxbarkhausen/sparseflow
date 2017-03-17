import functools
import tensorflow as tf
import numpy as np
import time


def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with self.graph.as_default():
                with tf.name_scope(function.__name__):
                    setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def timer(func):
    def decorator(*args, **kwargs):
        start = time.clock()
        for _ in range(0, 10):
            func(*args, **kwargs)
        end = time.clock()
        return end - start
    return decorator


def log_loss(true_pred, pred, loss_weights):
    epsilon = np.float(1e-6)
    pred = tf.maximum(epsilon, pred)
    pred = tf.minimum(1 - epsilon, pred)
    return tf.reduce_mean(loss_weights*(-true_pred * tf.log(pred)) -
                          ((1 - true_pred) * tf.log(1 - pred)))


def calculate_penalty(self, parameters_list):
    penalty = self.alpha * tf.contrib.layers.apply_regularization(
        self.regularization_function, parameters_list)
    return penalty


def early_validation_stopping(self, val_loss):
    if val_loss - self.last_loss > 0.1:
        self.stopping_counter += 1
        if self.stopping_counter == 20:
            message = "Stopping early, validation loss increased five times between batches."
            return message
    else:
        self.last_loss = val_loss
        return None


def convergence_stopping(self, train_loss):
    if abs(self.last_loss - train_loss) < 1e-10:
        self.stopping_counter += 1
        if self.stopping_counter == 5:
            message = "Stopping early, training loss changed by less than 1e-10."
            return message
    else:
        self.last_loss = train_loss
        return None


def to_array(data):
    try:
        data = data.toarray()
    except:
        pass
    return data
