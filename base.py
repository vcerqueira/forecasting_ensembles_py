import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from learning import get_regression_model
from utils import dict_subset_by_key
from metrics import mase


class Ensemble(RegressorMixin, BaseEstimator):
    """ Base Ensemble
    """

    def __init__(self, base_learners, omega: float = 0.5):

        self.base_fitted = False
        self.X = None
        self.y = None
        self.base_learners = base_learners
        self.base_learners_after_pruning = base_learners

        self.base_models = None
        self.base_hat_in = None
        self.base_loss = None
        self.omega = omega

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fitting the ensemble
        """

        self.create_base_models(X, y)
        self.base_predict_in_sample()

        return self

    def predict(self, X):
        """
        :param X:
        :return:
        """
        y_hat = self.predict_base_layer(X)

        average_y_hat = y_hat.mean(axis=1).values

        return average_y_hat

    def score(self, X, y, sample_weight=None):
        y_hat = self.predict(X)

        loss = mase(y_test=y, y_pred=y_hat, y_train=self.y)

        return loss

    def create_base_models(self, X, y, optimize_each_model: bool = False):
        """ Create base models

        :param X:
        :param y:
        :param optimize_each_model:
        :return:
        """
        base_models = dict()
        for method in self.base_learners:
            print(method)

            wf = get_regression_model(method)
            wf.fit(X=X, y=y)

            base_models[method] = wf

        self.base_models = base_models
        self.base_fitted = True
        self.X = X
        self.y = y

    def base_predict_in_sample(self):
        assert self.base_fitted

        y_hat_in = dict()
        for method in self.base_models:
            y_hat_in[method] = \
                self.base_models[method].predict(self.X).flatten()

        self.base_hat_in = pd.DataFrame(y_hat_in)
        self.base_loss = self.base_hat_in.apply(func=lambda x: x - self.y, axis=0)

    def prune_models(self):
        """
        Prune 1-self.omega % of the models based on training performance
        NOTE: make sure order of models stays equivalent
        :return: self
        """

        n_models = int(self.omega * self.base_loss.shape[1])

        model_mae = self.base_loss.abs().mean()
        model_sorted_rank = model_mae.rank().sort_values()
        models_to_keep = model_sorted_rank[:n_models].index.values

        self.base_learners_after_pruning = models_to_keep
        self.base_models = dict_subset_by_key(self.base_models, models_to_keep)
        self.base_hat_in = self.base_hat_in[models_to_keep]
        self.base_loss = self.base_loss[models_to_keep]

    def predict_base_layer(self, X):
        base_predictions = dict()
        for method in self.base_models:
            print(method)
            y_hat = self.base_models[method].predict(X)
            base_predictions[method] = y_hat.flatten()

        base_predictions = pd.DataFrame(base_predictions)

        return base_predictions

    def add_base_model(self):
        # todo Implement adding base model to ensemble

        return self

    def add_xy(self):
        # todo Implement adding observation to ensemble

        return self