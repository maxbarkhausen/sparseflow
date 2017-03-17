"""Base class for estimators."""
import tensorflow as tf
from ..functions import define_scope, to_array, log_loss
from abc import ABCMeta, abstractmethod, abstractproperty
from six.moves import cPickle as Pickle
from datetime import datetime
from sklearn.metrics import roc_auc_score


class BaseEstimator(metaclass=ABCMeta):
    def __init__(self, log_dir='log/', logging_frequency=10, negatives_ratio=1):
        self.log_dir = log_dir
        self.logging_frequency = logging_frequency
        self.last_loss = float("inf")
        self.best_val_loss = float("inf")
        self.negatives_ratio = negatives_ratio
        self._feature_indices = None
        self._graph = tf.Graph()
        self._num_features = None

    @property
    def graph(self):
        return self._graph

    @abstractproperty
    def model_params_dict(self):
        pass

    def init_model(self,):
        try:
            self.load_model_parameters()
        except:
            pass
        self._init_model()
        self._init_metrics()
        self.summary_writer_train.add_graph(self.graph)
        self.summary_writer_val.add_graph(self.graph)

    @abstractmethod
    def _init_model(self):
        pass

    def _init_metrics(self):
        with self.graph.as_default():
            self.roc_auc = tf.placeholder(tf.float32, ())
            tf.summary.scalar('training/val_roc_auc', self.roc_auc)
            self.streaming_roc_auc = tf.contrib.metrics.streaming_auc(self.predictions, self.y)
            tf.summary.scalar('streaming_roc_auc', self.streaming_roc_auc[0])

    @property
    def save_name(self):
        return self._save_name

    @save_name.setter
    def save_name(self, save_name):
        if save_name is not None:
            self._save_name = save_name
        else:
            self._save_name = datetime.now().strftime("%Y%m%d%H%M%S")

    @property
    def regularization_function(self):
        if self.penalty == 'l2':
            _regularization_function = tf.nn.l2_loss
        if self.penalty == 'l1':
            _regularization_function = tf.nn.l1_loss
        return _regularization_function

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, num_features):
        self._num_features = num_features

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=200,
            max_epochs=5000, early_stopping=False, logging_set_size=10000, verbose=True, save_name=None):
        self.save_name = save_name
        training_set_size, self.num_features = X_train.shape
        tf.reset_default_graph()
        self.init_model()
        session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            tf.global_variables_initializer().run(session=session)
            tf.local_variables_initializer().run(session=session)
        self._train(X_train, y_train, X_val, y_val, batch_size,
                    max_epochs, early_stopping, logging_set_size, training_set_size, session, verbose)
        self.save_model(session, 'last_epoch.ckpt')
        print("Model saved.")

    def _train(self, X_train, y_train, X_val, y_val,
               batch_size, max_epochs, early_stopping, logging_set_size,
               training_set_size, session, verbose):
        for step in range(max_epochs):
            batch_data, batch_labels = self._get_batch(X_train, y_train, step,
                                                       training_set_size, batch_size)
            self._train_step(batch_data, batch_labels, session)
            if step % int(max_epochs / self.logging_frequency) == 0 or step == max_epochs - 1:
                train_loss, val_loss = self._train_log(X_train, y_train, X_val, y_val,
                                                       step, logging_set_size, session, verbose)
                if early_stopping:
                    message = self.stopping_criterion(train_loss, val_loss)
                    if message is not None:
                        print(message)
                        self.save_model(session, 'last_epoch.ckpt')
                        print("Model saved.")
                        break
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.save_model(session, 'best_val.ckpt')
                    print("Validation loss improved, model saved.\n")
                    self.best_val_loss = val_loss

    @staticmethod
    def _get_batch(X_train, y_train, step, training_set_size, batch_size):
        offset = (step * batch_size) % (
            training_set_size - batch_size)
        batch_data = to_array(X_train[offset:(offset + batch_size)])
        batch_labels = y_train[offset:(offset + batch_size)]
        return batch_data, batch_labels

    def _train_step(self, batch_data, batch_labels, session):
        _, batch_loss = session.run([self.optimize, self.loss], feed_dict={
            self.X: batch_data, self.y: batch_labels})
        return batch_loss

    def _train_log(self, X_train, y_train, X_val, y_val, step, logging_set_size, session, verbose):
        val_loss, val_roc_auc = None, None
        train_loss, train_pred, train_roc_auc, streaming_roc_auc = self._write_summaries(
                                                                    to_array(X_train[0:logging_set_size]),
                                                                    y_train[0: logging_set_size],
                                                                    self.summary_writer_train,
                                                                    step, session)
        if X_val is not None:
            val_loss, val_pred, val_roc_auc, streaming_roc_auc = self._write_summaries(to_array(X_val),
                                                                                       y_val, self.summary_writer_val,
                                                                                       step, session)
        if verbose:
            print("Training loss at step ", step, ": ", train_loss)
            if val_loss is not None:
                print("Validation loss at step ", step, ": ", val_loss)
            print("Training roc_auc at step ", step, ": ", train_roc_auc)
            if val_roc_auc is not None:
                print("Validation roc_auc at step ", step, ": ", val_roc_auc)
            print("")
        return train_loss, val_loss

    def _write_summaries(self, X, y, summary_writer, step, session):
        if summary_writer == self.summary_writer_train:
            loss, predictions, streaming_roc_auc = session.run([self.loss, self.predictions,
                                                                self.streaming_roc_auc],
                                                               feed_dict={self.X: to_array(X), self.y: y})
        else:
            loss, predictions = session.run([self.loss, self.predictions], feed_dict={self.X: to_array(X), self.y: y})
            streaming_roc_auc = None
        roc_auc = roc_auc_score(y, predictions)
        summary = session.run(self.merged_summaries, feed_dict={
            self.X: X, self.y: y, self.roc_auc: roc_auc})
        summary_writer.add_summary(summary, step)
        return loss, predictions, roc_auc, streaming_roc_auc

    @define_scope
    def merged_summaries(self):
        _merged_summaries = tf.summary.merge_all()
        return _merged_summaries

    @define_scope
    def summary_writer_train(self):
        _summary_writer_train = tf.summary.FileWriter(
            self.log_dir + '/' + self.save_name + '/train/')
        return _summary_writer_train

    @define_scope
    def summary_writer_val(self):
        _summary_writer_val = tf.summary.FileWriter(
            self.log_dir + '/' + self.save_name + '/val/')
        return _summary_writer_val

    @abstractmethod
    def stopping_criterion(self, train_loss, val_loss):
        pass

    @define_scope
    def saver(self):
        self._saver = tf.train.Saver()
        return self._saver

    def save_model(self, session, save_name):
        self.saver.save(session, self.log_dir + self.save_name + '/' + save_name)
        with open(self.log_dir + self.save_name + '/params.dict', 'wb') as f:
            save = self.model_params_dict
            Pickle.dump(save, f, Pickle.HIGHEST_PROTOCOL)
            del save

    def restore_variables(self, save_name=None):
        session = tf.Session(graph=self.graph)
        if save_name is not None:
            try:
                self.saver.restore(session, self.log_dir + self.save_name + '/' + save_name)
                print("Restoring variables from ", save_name)
            except:
                print("Could not restore varibales from ", save_name)
        else:
            try:
                self.saver.restore(session, self.log_dir + self.save_name + '/' + 'best_val.ckpt')
                print("Restoring variables from save after best validation score.")
            except:
                print("Could not restore variables from save after best validation score, "
                      "trying to restore from last epoch.")
                try:
                    self.saver.restore(session, self.log_dir + self.save_name + '/' + 'last_epoch.ckpt')
                    print("Restoring variables from save after last epoch.")
                except:
                    print("Could not restore variables.")
        return session

    def load_model_parameters(self):
        with open(self.log_dir + self.save_name + '/params.dict', 'rb') as f:
            save = Pickle.load(f)
            for key, value in save.items():
                setattr(self, key, value)
            del save

    def predict_proba(self, X_pred):
        session = self.restore_variables()
        predictions = session.run(self.predictions, feed_dict={self.X: to_array(X_pred)})
        session.close()
        return predictions

    @abstractmethod
    def predictions(self):
        pass

    @define_scope
    def loss_weights(self):
        loss_weights = -(self.y-1)*(1/self.negatives_ratio)+self.y
        return loss_weights

    @define_scope
    def loss(self):
        loss = log_loss(self.y, self.predictions, self.loss_weights)
        tf.summary.scalar('loss', loss)
        return loss
