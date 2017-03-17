import numpy as np
from .conftest import sample_data
from ..estimators.pnn_estimator import PNNEstimator


def test_fit_and_predict():
    X_sample, y_sample, feature_indices = sample_data(num_samples=20000, num_features=1000)
    X_train = X_sample[0:15000]
    y_train = y_sample[0:15000]
    X_val = X_sample[15000:]
    y_val = y_sample[15000:]
    cf = PNNEstimator()
    cf.feature_indices = feature_indices
    cf.logging_frequency = 1
    cf.fit(X_train, y_train, X_val, y_val, max_epochs=100)
    sample_predictions = cf.predict_proba(X_val)
    assert sample_predictions.shape == (5000, 1)


def easy_classification_test(easy_classification_data):
    X, y, feature_indices = easy_classification_data
    cf = PNNEstimator()
    cf.feature_indices = feature_indices
    cf.logging_frequency = 1
    cf.fit(X, y, max_epochs=100)
    predictions = cf.predict_proba(X)
    assert np.mean(y - predictions < 0.01)
