import numpy as np
from .conftest import sample_data
from ..estimators.lr_estimator import LREstimator


def test_fit_and_predict():
    X_sample, y_sample, _ = sample_data(num_samples=20000, num_features=200)
    cf = LREstimator(learning_rate=0.01)
    cf.logging_frequency = 1
    cf.fit(X_sample, y_sample, max_epochs=100)
    sample_predictions = cf.predict_proba(X_sample)
    assert sample_predictions.shape == (20000, 1)


def easy_classification_test(easy_classification_data):
    X, y, _ = easy_classification_data
    cf = LREstimator()
    cf.logging_frequency = 1
    cf.fit(X, y, max_epochs=100)
    predictions = cf.predict_proba(X)
    assert np.mean(y - predictions < 0.01)
