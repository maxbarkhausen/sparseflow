import tensorflow as tf
import numpy as np
from pytest import approx
from .conftest import sample_data, sample_V
from ..estimators.fm_estimator import FMEstimator


def factorization_term_tf(X, V):
    cf = FMEstimator()
    cf.num_features = X.shape[1]
    tf.reset_default_graph()
    session = tf.Session(graph=cf.graph)
    cf._init_model()
    result = cf.factorization.eval(session=session,
                                   feed_dict={cf.X: X, cf.V: V})
    return result


def factorization_term_np_einsum(X, V):
    factors = np.triu(np.einsum('ij,ik->ijk', X, X), 1)
    vector_products = np.dot(V, np.transpose(V))
    result = np.sum(factors * vector_products, axis=(1, 2))
    return result


def test_factorization():
    X_sample, _, _ = sample_data(num_samples=200, num_features=40)
    V_sample = sample_V(num_features=40)
    ft1 = factorization_term_tf(X_sample, V_sample)
    ft2 = factorization_term_np_einsum(X_sample, V_sample)
    for i in range(0, ft1.shape[0]):
        assert ft1[i] == approx(ft2[i])


def test_fit_and_predict():
    X_sample, y_sample, _ = sample_data(num_samples=20000, num_features=1000)
    X_train = X_sample[0:15000]
    y_train = y_sample[0:15000]
    X_val = X_sample[15000:]
    y_val = y_sample[15000:]
    cf = FMEstimator()
    cf.logging_frequency = 1
    cf.fit(X_train, y_train, X_val, y_val, max_epochs=100)
    sample_predictions = cf.predict_proba(X_val)
    assert sample_predictions.shape == (5000, 1)


def easy_classification_test(easy_classification_data):
    X, y, _ = easy_classification_data
    cf = FMEstimator()
    cf.logging_frequency = 1
    cf.fit(X, y, max_epochs=100)
    predictions = cf.predict_proba(X)
    assert np.mean(y - predictions < 0.01)
