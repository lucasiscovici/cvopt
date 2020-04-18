### SOURCE FROM scipy-optimize


import numpy as np

from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from GPyOpt.models.base import  BOModel

def _parallel_fit(regressor, X, y):
    return regressor.fit(X, y)


class GradientBoostingQuantileRegressor(BaseEstimator, RegressorMixin):
    """Predict several quantiles with one estimator.
    This is a wrapper around `GradientBoostingRegressor`'s quantile
    regression that allows you to predict several `quantiles` in
    one go.
    Parameters
    ----------
    quantiles : array-like
        Quantiles to predict. By default the 16, 50 and 84%
        quantiles are predicted.
    base_estimator : GradientBoostingRegressor instance or None (default)
        Quantile regressor used to make predictions. Only instances
        of `GradientBoostingRegressor` are supported. Use this to change
        the hyper-parameters of the estimator.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `fit`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.
    """
    def __init__(self, quantiles=[0.16, 0.5, 0.84], base_estimator=None,
                 n_jobs=1, random_state=None):
        self.quantiles = quantiles
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit one regressor for each quantile.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape=(n_samples,)
            Target values (real numbers in regression)
        """
        rng = check_random_state(self.random_state)

        if self.base_estimator is None:
            base_estimator = GradientBoostingRegressor(loss='quantile')
        else:
            base_estimator = self.base_estimator

            if not isinstance(base_estimator, GradientBoostingRegressor):
                raise ValueError('base_estimator has to be of type'
                                 ' GradientBoostingRegressor.')

            if not base_estimator.loss == 'quantile':
                raise ValueError('base_estimator has to use quantile'
                                 ' loss not %s' % base_estimator.loss)

        # The predictions for different quantiles should be sorted.
        # Therefore each of the regressors need the same seed.
        base_estimator.set_params(random_state=rng)
        regressors = []
        for q in self.quantiles:
            regressor = clone(base_estimator)
            regressor.set_params(alpha=q)

            regressors.append(regressor)

        self.regressors_ = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_parallel_fit)(regressor, X, y)
            for regressor in regressors)

        return self

    def predict(self, X, return_std=False, return_quantiles=False):
        """Predict.
        Predict `X` at every quantile if `return_std` is set to False.
        If `return_std` is set to True, then return the mean
        and the predicted standard deviation, which is approximated as
        the (0.84th quantile - 0.16th quantile) divided by 2.0
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        predicted_quantiles = np.asarray(
            [rgr.predict(X) for rgr in self.regressors_])
        if return_quantiles:
            return predicted_quantiles.T

        elif return_std:
            std_quantiles = [0.16, 0.5, 0.84]
            is_present_mask = np.in1d(std_quantiles, self.quantiles)
            if not np.all(is_present_mask):
                raise ValueError(
                    "return_std works only if the quantiles during "
                    "instantiation include 0.16, 0.5 and 0.84")
            low = self.regressors_[self.quantiles.index(0.16)].predict(X)
            high = self.regressors_[self.quantiles.index(0.84)].predict(X)
            mean = self.regressors_[self.quantiles.index(0.5)].predict(X)
            return mean, ((high - low) / 2.0)

        # return the mean
        return self.regressors_[self.quantiles.index(0.5)].predict(X)
    
class GBRTModel(BOModel):
    """
    General class for handling a GBRT in GPyOpt.
    .. Note:: The model has beed wrapper 'as it is' from  Scikit-learn. Check
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    for further details.
    """

    analytical_gradient_prediction = False

    def __init__(self, quantiles=[0.16, 0.5, 0.84], base_estimator=None,
                 n_jobs=1, random_state=None):

        self.quantiles = quantiles
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.model = None

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        self.X = X
        self.Y = Y
        self.model = GradientBoostingQuantileRegressor(quantiles =self. quantiles,
                                        random_state = self.random_state,
                                        base_estimator = self.base_estimator,
                                        n_jobs = self.n_jobs)

        self.model.fit(X,Y.flatten())


    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        self.X = X_all
        self.Y = Y_all
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.fit(X_all, Y_all.flatten())

    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X.
        """
        rep=self.model.predict(X,return_std=True)
        return np.reshape(rep[0],(-1,1)),np.reshape(rep[1],(-1,1))

    def get_fmin(self):
        return self.model.predict(self.X).min()
    
    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(list(self.model.get_params().values()))
    
    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return list(self.model.get_params().keys())
