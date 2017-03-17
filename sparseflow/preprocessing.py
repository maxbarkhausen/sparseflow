"""A class for preprocessing impressions datasets and pickling training sets."""
from six.moves import cPickle as Pickle
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import re


class Preprocessor:
    """Some functions for preprocessing impression datasets in one convenient class."""

    def __init__(self):
        self.categorical_features = None
        self.time_features = None
        self.quantitative_features = None
        pass

    def extract_data(self, csv_files=[], csv_directories=[], save_file=False):
        import os
        for directory in csv_directories:
            files = os.listdir(directory)

            def is_csv(f):
                if f.endswith('csv'):
                    return True
                else:
                    return False

            files = [directory + '/' + f for f in files if is_csv(f)]
            csv_files += files
        datasets = []
        for f in csv_files:
            datasets.append(pd.DataFrame.from_csv(f))
            data = pd.concat(datasets)
        self.raw_data = data
        return data

    def make_dataset(self, y_label, quantitative_features=None, categorical_features=None,
                     time_features=None, input_data=None):
        if input_data is not None:
            self.raw_data = input_data
        if time_features is not None:
            self.time_features = self.make_categorical_time_features(time_features)
        if categorical_features is not None or time_features is not None:
            self.categorical_features, self.feature_indices = self.encode_categorical_features(categorical_features)
        if quantitative_features is not None:
            self.quantitative_features = self.normalize_quantitative_features(quantitative_features)
        self.X = sparse.hstack((self.categorical_features, self.quantitative_features))
        self.y = self.raw_data[y_label].values.transpose()
        return self.X, self.y, self.feature_indices

    def encode_categorical_features(self, categorical_features):
        if categorical_features is not None:
            self.categorical_features = self.raw_data[categorical_features].fillna(-999).astype(str)
            if self.time_features is not None:
                self.categorical_features = pd.concat([
                    self.categorical_features, self.time_features], axis=1)
        else:
            self.categorical_features = self.time_features
        self.categorical_features = self.categorical_features.apply(LabelEncoder().fit_transform).values
        self.ohe = OneHotEncoder(sparse=True)
        self.categorical_features = self.ohe.fit_transform(self.categorical_features)
        return self.categorical_features, self.ohe.feature_indices_

    def normalize_quantitative_features(self, quantitative_features):
        features_to_normalize = self.raw_data[quantitative_features].values
        normalized_features = normalize(features_to_normalize, axis=0)
        self.quantitative_features = sparse.csr_matrix(normalized_features)
        return self.quantitative_features

    def make_categorical_time_features(self, time_features):
        encoded_features = []
        for feature in time_features:
            self.raw_data[feature] = self.raw_data[feature].apply(pd.to_datetime)
            weekday = self.raw_data[feature].dt.dayofweek
            hour = self.raw_data[feature].dt.hour
            encoded_features.append(pd.concat([weekday, hour], axis=1))
        self.time_features = pd.concat(encoded_features, axis=1)
        return self.time_features

    def convert_banner_size(self, column):

        def calc_size(string_size):
            numbers = re.findall(r"[0-9]+", string_size)
            total = 1
            for number in numbers:
                total *= np.float(number)
            return total

        self.raw_data[column] = self.raw_data[column].apply(calc_size)

    @staticmethod
    def train_validation_test_split(validation_set_size, test_set_size, X, y):
        assert validation_set_size + test_set_size < 1, "Size of validation set+test set > 1"
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=validation_set_size + test_set_size,
                                                            random_state=103)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size=validation_set_size / (
                                                            validation_set_size + test_set_size))
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def load_data(load_list, pickle_file):
        with open(pickle_file, 'rb') as f:
            save = Pickle.load(f)
            load_items = []
            for item in load_list:
                load_items.append(save[item])
            del save
        return load_items

    @staticmethod
    def save_data(save_dict, pickle_file):
        with open(pickle_file, 'wb') as f:
            save = save_dict
            Pickle.dump(save, f, Pickle.HIGHEST_PROTOCOL)
            del save
