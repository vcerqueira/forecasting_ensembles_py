import numpy as np
import pandas as pd

from utils \
    import (normalize_and_proportion,
            proportion)

from learning import get_regression_model
from base import Ensemble


class ADE(Ensemble):
    """ ADE
    todo: docs
    """

    def __init__(self,
                 meta_learner,
                 base_learners,
                 lambda_: int,
                 omega: float = 0.5):

        """
        # todo: sequential re-weighting
        :param meta_learner:
        :param base_learners:
        :param lambda_:
        :param omega:
        """

        super().__init__(base_learners=base_learners,
                         omega=omega)

        self.meta_learner = meta_learner
        self.meta_models = None
        self.lambda_ = lambda_

    def fit(self, X: np.ndarray, y: np.ndarray):

        print("Fitting base models")
        self.create_base_models(X, y)
        print("Computing in loss")
        self.base_predict_in_sample()
        print("Fitting meta models")
        self.create_meta_models()

        return self

    def predict(self, X):

        meta_hat, W = self.predict_meta_layer(X)
        base_hat = self.predict_base_layer(X)

        iter_data = np.arange(X.shape[0])

        y_hat_ade = []
        for i in iter_data:
            models_to_keep = \
                self.dynamic_pruning(meta_hat.iloc[:(i + 1), :])

            W_i = proportion(W.loc[i, models_to_keep].values)
            y_hat_i = base_hat.loc[i, models_to_keep].values

            y_hat_f = np.sum(y_hat_i * W_i)

            y_hat_ade.append(y_hat_f)

        return np.array(y_hat_ade)

    def create_meta_models(self):
        """
        Creating meta-models
        """

        meta_models = dict()
        for method in self.base_models:
            err_method = self.base_loss[method].values

            wf_meta = get_regression_model(self.meta_learner)
            wf_meta.fit(self.X, y=err_method)

            meta_models[method] = wf_meta

        self.meta_models = meta_models

    def dynamic_pruning(self, meta_prediction):
        """Dynamic pruning based on meta-layer predictions
        todo: dynamic pruning should be based on actual loss

        :param meta_prediction:
        :return:
        """
        n_models = int(self.omega * self.base_loss.shape[1])

        known_base_loss = \
            pd.concat([self.base_loss.abs(), meta_prediction],
                      ignore_index=True)

        avg_loss_lambda = \
            known_base_loss.iloc[-self.lambda_:, :].mean()

        model_sorted_rank = avg_loss_lambda.rank().sort_values()
        models_to_keep = model_sorted_rank[:n_models].index.values

        return models_to_keep

    def predict_meta_layer(self, X):
        meta_predictions = dict()
        for method in self.meta_models:
            raw_meta_hat = self.meta_models[method].predict(X)
            abs_meta_hat = [np.abs(x) for x in raw_meta_hat]
            meta_predictions[method] = abs_meta_hat

        meta_predictions = pd.DataFrame(meta_predictions)

        W = meta_predictions.apply(
            func=lambda x: normalize_and_proportion(-x),
            axis=1)

        return meta_predictions, W
