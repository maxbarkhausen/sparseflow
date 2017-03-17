import tensorflow as tf
from math import sqrt


def xavier_weights(shape):
    return tf.Variable(tf.random_uniform(shape=shape, minval=-sqrt(6) / sqrt(shape[0] + shape[1]),
                                         maxval=sqrt(6) / sqrt(shape[0] + shape[1])))


def biases(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
