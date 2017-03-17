# Sparseflow

Library for two-class probability estimation with extremely sparse data, using tensorflow. It currently implements several estimators for two-class probability estimation: a logistic regression model, a factorization machine, and two deep learning models that take embeddings of data fields consisting of several binary features (as opposed to single binary features) as their input layer. Further models and features will be added in the future.

This is a python 3 library.

## Installation

First create an Anaconda environment.

    $ conda create --name sparseflow python=3

Activate the conda environment:

    $ source activate sparseflow

Clone `sparseflow` repository to current directory:

    $ git clone https://github.com/maxbarkhausen/sparseflow.git

Install all requirements:

    $ conda install --file requirements.txt --yes
    $ pip install -r requirements_pip.txt

Now install the `sparseflow` package:

    $ python setup.py develop

## Use

### Preprocessing

The preprocessing functions are implemented in preprocessing.py, which contains a single class called Preprocessor.

Examples:

    from preprocessing import Preprocessor


    prep = Preprocessor()
    dataframe = prep.extract_data(cvs_directories = ['data1', data2'])

    cf = ['user_ip', 'country', 'language']
    qf = ['age', 'size']
    tf = ['datetime']

    X, y , feature_indices = prep.make_dataset(categorical_features = cf, quantitative_features = qf, time_features = tf, input = dataframe)
    X_train, y_train, X_val, y_val, X_test, y_test = train_validation_test_split(X, y, test_set_size=0.1)

### Estimators

Four estimators are currently implemented in the estimators module: a simple logistic regression model with optional l1/l2-regularization, a factorization machine based on Rendle 2010, and two deep learning models that take as their first layer field-wise dense embeddings of the one-hot encoded feature-vectors, inspired by recent work in NLP. These are implemented in the classes LREstimator, FMEstimator, PNNEstimator and ENNEstimator, contained in separate files. 
 
 Example 1:
 
    from estimators.enn_estimator import ENNEstimator
    
    
    enn = ENNEstimator()
    enn.feature_indices = feature_indices
    enn.log_dir = 'log/enn/'
    enn.fit(X_train, y_train, X_val, y_val, max_epochs = 20000, save_name = 'run1')
    predictions = enn.predict_proba(X_test)
    
 Example 2:
 
    from estimators.ennestimator import ENNEstimator
    
    
    enn = ENNEstimator()
    enn.log_dir = 'log/enn'
    enn.re(save_name = 'run1')
    enn.restore_variables()
    enn.predict_proba(X_test2)

### Logging

All models automatically log several metrics for a subsample of the training set and validation set, if provided as an argument to the fit method. These, as well as visualization of the tensorflow graphs, can be accessed via tensorboard. This is useful especially for quickly comparing the performance of different models on a dataset.

Example:

    $tensorboard --logdir='log/enn'





