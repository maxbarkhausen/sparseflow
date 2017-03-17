"""Module for testing helper functions."""
from pytest import approx
import numpy as np
import tensorflow as tf
from ..functions import log_loss


def test_log_loss(sample_predictions):
    tf.reset_default_graph()
    session = tf.Session()
    y_true, y_pred = sample_predictions

    def calc_log_loss(true_pred, pred):
        log_loss = (-true_pred * np.log(pred)) - (
            (1 - true_pred) * np.log(1 - pred))
        return log_loss

    calc_log_losses = np.vectorize(calc_log_loss)
    average_loss = np.mean(calc_log_losses(y_true, y_pred))
    test_average_loss = session.run(log_loss(y_true, y_pred, 1))
    session.close()

    assert test_average_loss == approx(average_loss)
