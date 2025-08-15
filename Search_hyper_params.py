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

class Serch_hyperParams:
    """
    Class for searching hyperparameters
    folds: list of folds for cross-validation
    mode: 'reg' for regression, 'cls' for classification
    """
    def __init__(self, train, val, mode='reg', use_gpu=False):
        self.train_splits_of_splitters = train
        self.valid_splits_of_splitters = val
        self.mode = mode
        self.use_gpu = use_gpu
        self.records = {}
        self.params = {}

    def search_in_xgboost(self,n=10):
        """
        Search hyperparameters for XGBoost model using Optuna.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0, step=0.1),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': 42,
                'device': 'gpu'  # Use 'gpu' if you have a compatible GPU and want to use it
            }
            model = XGBRegressor(**params)
            scores = []
            for tr, va in zip(self.train_splits_of_splitters[0], self.valid_splits_of_splitters[0]):
                model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])], verbose=False)
                score = rmse(va['y'].to_numpy(), model.predict(va['X']))
                scores.append(score)
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n)
        self.records['xgboost'] = study.best_value
        self.params['xgboost'] = study.best_params
        return study.best_params

    def search_in_randomforest(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            model = RandomForestRegressor(**params)
            scores = []
            for tr, va in zip(self.train_splits_of_splitters[0], self.valid_splits_of_splitters[0]):
                model.fit(tr['X'], tr['y'])
                score = rmse(va['y'].to_numpy(), model.predict(va['X']))
                scores.append(score)
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n, n_jobs=4)
        self.records['randomforest'] = study.best_value
        self.params['randomforest'] = study.best_params
        return study.best_params

    def search_in_catboost(self, n=10):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, step=0.5),
                'random_state': 42,
                'verbose': False
            }
            model = CatBoostRegressor(**params)
            scores = []
            for tr, va in zip(self.train_splits_of_splitters[0], self.valid_splits_of_splitters[0]):
                model.fit(tr['X'], tr['y'], eval_set=(va['X'], va['y']), verbose=False)
                score = rmse(va['y'].to_numpy(), model.predict(va['X']))
                scores.append(score)
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n)
        self.records['catboost'] = study.best_value
        self.params['catboost'] = study.best_params
        return study.best_params

    def search_in_extratrees(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            model = ExtraTreesRegressor(**params)
            scores = []
            for tr, va in zip(self.train_splits_of_splitters[0], self.valid_splits_of_splitters[0]):
                model.fit(tr['X'], tr['y'])
                score = rmse(va['y'].to_numpy(), model.predict(va['X']))
                scores.append(score)
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n, n_jobs=4)
        self.records['extratrees'] = study.best_value
        self.params['extratrees'] = study.best_params
        return study.best_params

    def search_in_lgbm(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100, step=10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                'random_state': 42,
                'verbose': -1  # Suppress output
            }
            model = LGBMRegressor(**params)
            scores = []
            for tr, va in zip(self.train_splits_of_splitters[0], self.valid_splits_of_splitters[0]):
                model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])])
                score = rmse(va['y'].to_numpy(), model.predict(va['X']))
                scores.append(score)
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n)
        self.records['lgbm'] = study.best_value
        self.params['lgbm'] = study.best_params
        return study.best_params