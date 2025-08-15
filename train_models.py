from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
# lgbm
from lightgbm import LGBMRegressor
#catboost
from catboost import CatBoostRegressor
#MLP
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error as rmse
from tqdm import tqdm
import optuna
import numpy as np

class Training_models:
    def __init__(self, train, val,params=None):
        self.train_splits = train
        self.valid_splits = val
        self.params = params if params else {}

        self.models = {
            'xgboost': XGBRegressor(**self.params.get('xgboost', {})),
            'randomforest': RandomForestRegressor(**self.params.get('randomforest', {})),
            'catboost': CatBoostRegressor(**self.params.get('catboost', {})),
            'extratrees': ExtraTreesRegressor(**self.params.get('extratrees', {})),
            'lgbm': LGBMRegressor(**self.params.get('lgbm', {})),
        }

        self.trained_models = {}

        self.scores = {}

    def train_models(self):
        """
        Train all models on the provided training splits.
        """
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            models = []
            scores = []
            for tr, va in zip(self.train_splits[0], self.valid_splits[0]):
                try:
                    model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])], verbose=False)
                except:
                    try:
                        model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])])
                    except:
                        model.fit(tr['X'], tr['y'])
                models.append(model)
                score = rmse(va['y'].to_numpy(), model.predict(va['X']))
                scores.append(score)
            self.scores[model_name] = np.mean(scores)
            self.trained_models[model_name] = models
            print(f"{model_name} trained with RMSE: {self.scores[model_name]}")