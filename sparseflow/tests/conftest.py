import random
import pytest
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from ..estimators.fm_estimator import FMEstimator


@pytest.fixture(scope='session')
def sample_dataset():
    columns = ['auction_ID', 'country', 'is_click', 'size', 'date']
    sample_data = [
        [516, "Ger", 0, "200x300", "2014-10-18 2:31:12"],
        [730, "BR", 0, "30X20", "2014-10-18 4:31:12"],
        [345, "FR", 1, "44X99", "2012-11-18 6:31:12"],
        [630, "USA", 0, "100X400", "2013-10-18 8:31:12"],
        [369, "FR", 0, "420X50", "2015-10-18 15:31:12"]
    ]
    sample_dataset = pd.DataFrame(sample_data, columns=columns)
    assert (sample_dataset.columns.tolist() == columns)
    return sample_dataset


def factorization_term_tf(X, V):
    cf = FMEstimator()
    cf.num_features = X.shape[1]
    cf._init_model()
    result = cf.factorization.eval(session=cf.session,
                                   feed_dict={cf.X: X, cf.V: V})
    return result


@pytest.fixture(scope='session')
def sample_datafile(tmpdir_factory, sample_dataset):
    sample_data = sample_dataset
    tmp_dir = tmpdir_factory.mktemp('temp_data')
    test_file = tmp_dir.join('test_data.csv')
    sample_data.to_csv(str(test_file))
    reloaded_sample_data = pd.read_csv(test_file, index_col=0)
    assert (len(tmp_dir.listdir()) == 1)
    assert (sample_data.columns.tolist() == reloaded_sample_data.columns.tolist())
    return test_file


@pytest.fixture
def sample_predictions():
    y_pred = np.random.random_sample(10)
    y_true = np.random.randint(0, 2, 10)
    return y_true, y_pred


def sample_data(num_samples=200, num_features=2000):
    X = np.random.randint(0, 2, num_samples * num_features). \
        reshape(num_samples, num_features).astype(float)
    y = np.random.randint(0, 2, num_samples).reshape(num_samples, 1).astype(float)
    feature_indices = np.array(sorted(random.sample(range(0, num_features), int(num_features / 10))))
    return X, y, feature_indices


def sample_V(num_features=2000, num_dims=8):
    V = np.random.random_sample((num_features, num_dims))
    return V


@pytest.fixture
def easy_classification_data():
    X_pos = np.ones(shape=(1000, 100))
    X_neg = np.zeros(shape=(1000, 100))
    X = np.vstack([X_pos, X_neg]).astype(float)
    y_pos = np.ones(shape=(1000, 1))
    y_neg = np.zeros(shape=(1000, 1))
    y = np.vstack([y_pos, y_neg]).astype(float)
    X, y = shuffle(X, y)
    feature_indices = np.array(sorted(random.sample(range(0, 100), 10)))
    return X, y, feature_indices
