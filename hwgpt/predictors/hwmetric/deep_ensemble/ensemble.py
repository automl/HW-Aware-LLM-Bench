import xgboost as xgb

import json
import logging
import os
import sys
import numpy as np

from sklearn.metrics import mean_squared_log_error
from typing import List
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def get_ensemble_members(model_type="xgboost"):
    ensemble_list = []
    if model_type == "xgboost":
        for depth in [3, 4, 5]:
            for n_estimators in [100, 200, 300]:
                for learning_rate in [0.01, 0.001, 0.3]:
                    ensemble_list.append(
                        xgb.XGBRegressor(
                            max_depth=depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            n_jobs=-1,
                        )
                    )
    elif model_type == "lightgbm":
        for depth in [2, 4, 8]:
            for n_estimators in [300, 400, 500]:
                for learning_rate in [0.01, 0.001, 0.3]:
                    ensemble_list.append(
                        LGBMRegressor(
                            max_depth=depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            n_jobs=-1,
                            num_leaves=2 ** depth,
                        )
                    )
    elif model_type == "mix":
        # define all the models
        ensemble_list = [
            xgb.XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.01, n_jobs=-1),
            xgb.XGBRegressor(max_depth=4, n_estimators=200, learning_rate=0.001, n_jobs=-1),
            xgb.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.3, n_jobs=-1),
            xgb.XGBRegressor(max_depth=8, n_estimators=100, learning_rate=0.01, n_jobs=-1),
            
            LGBMRegressor(max_depth=2, n_estimators=300, learning_rate=0.01, n_jobs=-1, num_leaves=2 ** 2),
            LGBMRegressor(max_depth=4, n_estimators=400, learning_rate=0.001, n_jobs=-1, num_leaves=2 ** 4),
            LGBMRegressor(max_depth=8, n_estimators=500, learning_rate=0.3, n_jobs=-1, num_leaves=2 ** 8),
            LGBMRegressor(max_depth=3, n_estimators=100, learning_rate=0.01, n_jobs=-1, num_leaves=2 ** 3),
            LinearRegression(),
            Ridge(),
            RandomForestRegressor(),
        ]
    
    # else:
    #    raise ValueError("Member not implemented")
    return ensemble_list


class BaggingEnsemble:
    def __init__(self, member_model_type, ensemble_size):
        self.ensemble_size = ensemble_size
        self.ensemble = get_ensemble_members("mix")

    def save(self):
        raise NotImplementedError

    def load(self, model_paths):
        raise NotImplementedError

    def train(self, X: np.array, y: np.array | List[np.array]):
        for i, regressor in enumerate(self.ensemble):
            if isinstance(X, List):
                regressor.fit(X, y[i])
            else:
                regressor.fit(X, y)

    def validate(self, X: np.array, y: np.array | List[np.array]):
        predictions = []
        for i, regressor in enumerate(self.ensemble):
            print(regressor.predict(X).shape)
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        print(all_predictions.shape)
        all_predictions_mean = np.mean(all_predictions, axis=-1)
        print(all_predictions_mean.shape)
        all_predictions_std = np.std(all_predictions, axis=-1)
        print(all_predictions_std.shape)
        msle_mean = np.sqrt(mean_squared_log_error(y, all_predictions_mean))
        msle_std = None
        return msle_mean, msle_std

    def predict(self, X: np.array):
        predictions = []
        for i, regressor in enumerate(self.ensemble):
            regressor.predict(X)
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        mean = np.mean(all_predictions, axis=-1)
        std = np.std(all_predictions, axis=-1)
        noisy_predictions = [
            np.random.normal(loc=m, scale=s, size=1)[0] for m, s in zip(mean, std)
        ]
        return mean, std, noisy_predictions
