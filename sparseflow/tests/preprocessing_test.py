"""Module for testing the preprocessing functions."""
import numpy as np
import os
from ..preprocessing import Preprocessor

Prep = Preprocessor()


def binary_array(array):

    def is_zero_or_one(x):
        if x == 1 or x == 0:
            return True
        else:
            return False

    zeroone = np.vectorize(is_zero_or_one)
    if zeroone(array).any():
        return True
    else:
        return False


def test_extract_data(sample_datafile, sample_dataset):
    sample_file = sample_datafile
    sample_data = sample_dataset
    data_directory = os.path.dirname(str(sample_file))
    from_directory_data = Prep.extract_data(csv_directories=[data_directory])
    from_file_data = Prep.extract_data(csv_files=[sample_file])
    assert (from_file_data.equals(sample_data))
    assert (from_directory_data.equals(sample_data))


def test_convert_banner_size(sample_datafile):
    sample_file = sample_datafile
    data = Prep.extract_data(csv_files=[sample_file])
    Prep.convert_banner_size('size')
    assert (data['size'].iloc[0] == 60000)
    assert (data['size'].dtype == np.float64)


def test_make_dataset():
    categorical_features = ['country']
    quantitative_features = ['size']
    time_features = ['date']
    y_label = 'is_click'
    X, y, feature_indices = Prep.make_dataset(y_label=y_label,
                                              quantitative_features=quantitative_features,
                                              categorical_features=categorical_features,
                                              time_features=time_features)
    size_argmax = np.argmax(X.toarray()[:, -1])
    assert size_argmax == 0
    size_argmin = np.argmin(X.toarray()[:, -1])
    assert size_argmin == 1
    assert np.max(X.toarray()[:, -1]) < 1
    assert np.min(X.toarray()[:, -1]) > 0
    assert np.array_equal([0, 0, 1, 0, 0], y)
    assert binary_array(X.toarray()[:, 0:-2])


def test_train_validation_test_split():
    X = Prep.X
    y = Prep.y
    X_train, y_train, X_val, y_val, X_test, y_test = Prep.train_validation_test_split(0.1, 0.1, X, y)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == X.shape[1]


def test_save_data_and_load_data():
    Prep.save_data(save_dict={'X': Prep.X, 'y': Prep.y}, pickle_file='test_data.pickle')
    X, y = Prep.load_data(['X', 'y'], 'test_data.pickle')
    assert np.array_equal(X.toarray(), Prep.X.toarray())
    assert np.array_equal(y, Prep.y)
    os.remove('test_data.pickle')
