import numpy as np
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import root_mean_squared_error as rmse
from xgboost import XGBClassifier, XGBRegressor


class Serch_hyperParams:
    """
    Class for searching hyperparameters.

    mode: 'reg' for regression, 'class' for classification.
    """
    def __init__(self, train, val, mode='reg', use_gpu=False):
        self.train_splits_of_splitters = train
        self.valid_splits_of_splitters = val
        self.mode = mode
        self.use_gpu = use_gpu
        self.records = {}
        self.params = {}
        if self.mode not in ['reg', 'class']:
            raise ValueError("Mode must be either 'reg' or 'class'.")

    def _score_model(self, model):
        scores = []
        for tr, va in zip(self.train_splits_of_splitters[0], self.valid_splits_of_splitters[0]):
            try:
                model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])], verbose=False)
            except TypeError:
                try:
                    model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])])
                except TypeError:
                    model.fit(tr['X'], tr['y'])
            except Exception:
                model.fit(tr['X'], tr['y'])

            predictions = model.predict(va['X'])
            if self.mode == 'class':
                scores.append(accuracy_score(va['y'].to_numpy(), predictions))
            else:
                scores.append(rmse(va['y'].to_numpy(), predictions))
        return float(np.mean(scores))

    def _finish_study(self, study, name):
        self.records[name] = float(study.best_value)
        self.params[name] = study.best_params
        return study.best_params

    def search_in_xgboost(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0, step=0.1),
                'random_state': 42,
            }
            if self.use_gpu:
                params['device'] = 'cuda'
            if self.mode == 'class':
                model = XGBClassifier(eval_metric='logloss', **params)
                return self._score_model(model)
            model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **params)
            return self._score_model(model)

        direction = 'maximize' if self.mode == 'class' else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n)
        return self._finish_study(study, 'xgboost')

    def search_in_randomforest(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 24),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1,
            }
            model_cls = RandomForestClassifier if self.mode == 'class' else RandomForestRegressor
            return self._score_model(model_cls(**params))

        direction = 'maximize' if self.mode == 'class' else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n, n_jobs=1)
        return self._finish_study(study, 'randomforest')

    def search_in_catboost(self, n=10):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 800, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, step=0.5),
                'random_state': 42,
                'verbose': False,
            }
            if self.mode == 'class':
                model = CatBoostClassifier(loss_function='Logloss', **params)
            else:
                model = CatBoostRegressor(**params)
            return self._score_model(model)

        direction = 'maximize' if self.mode == 'class' else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n)
        return self._finish_study(study, 'catboost')

    def search_in_extratrees(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 24),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1,
            }
            model_cls = ExtraTreesClassifier if self.mode == 'class' else ExtraTreesRegressor
            return self._score_model(model_cls(**params))

        direction = 'maximize' if self.mode == 'class' else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n, n_jobs=1)
        return self._finish_study(study, 'extratrees')

    def search_in_lgbm(self, n=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100, step=10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                'random_state': 42,
                'verbose': -1,
            }
            model_cls = LGBMClassifier if self.mode == 'class' else LGBMRegressor
            return self._score_model(model_cls(**params))

        direction = 'maximize' if self.mode == 'class' else 'minimize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n)
        return self._finish_study(study, 'lgbm')
