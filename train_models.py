import copy

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import root_mean_squared_error as rmse
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor


class Training_models:
    def __init__(self, train, val, params=None, mode='reg'):
        self.train_splits = train
        self.valid_splits = val
        self.params = params if params else {}
        self.mode = mode
        if self.mode not in ['reg', 'class']:
            raise ValueError("Mode must be either 'reg' or 'class'.")
        self.models = self._init_models()
        self.trained_models = {}
        self.scores = {}

    def _model_params(self, name):
        return copy.deepcopy(self.params.get(name, {}))

    def _n_classes(self):
        if self.mode != 'class':
            return None
        y = self.train_splits[0][0]['y']
        return int(np.unique(np.asarray(y)).size)

    def _init_models(self):
        if self.mode == 'class':
            n_classes = self._n_classes()
            xgboost_params = self._model_params('xgboost')
            xgboost_params.setdefault('eval_metric', 'mlogloss' if n_classes > 2 else 'logloss')
            if n_classes > 2:
                xgboost_params.setdefault('objective', 'multi:softprob')
                xgboost_params.setdefault('num_class', n_classes)

            catboost_params = self._model_params('catboost')
            catboost_params.setdefault('loss_function', 'MultiClass' if n_classes > 2 else 'Logloss')

            return {
                'xgboost': XGBClassifier(
                    random_state=42,
                    **xgboost_params,
                ),
                'randomforest': RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **self._model_params('randomforest'),
                ),
                'catboost': CatBoostClassifier(
                    random_state=42,
                    verbose=False,
                    **catboost_params,
                ),
                'extratrees': ExtraTreesClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **self._model_params('extratrees'),
                ),
                'lgbm': LGBMClassifier(
                    random_state=42,
                    verbose=-1,
                    **self._model_params('lgbm'),
                ),
            }

        return {
            'xgboost': XGBRegressor(
                random_state=42,
                objective='reg:squarederror',
                eval_metric='rmse',
                **self._model_params('xgboost'),
            ),
            'randomforest': RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                **self._model_params('randomforest'),
            ),
            'catboost': CatBoostRegressor(
                random_state=42,
                verbose=False,
                **self._model_params('catboost'),
            ),
            'extratrees': ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1,
                **self._model_params('extratrees'),
            ),
            'lgbm': LGBMRegressor(
                random_state=42,
                verbose=-1,
                **self._model_params('lgbm'),
            ),
        }

    def _fit_model(self, model, tr, va):
        try:
            model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])], verbose=False)
        except TypeError:
            try:
                model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])])
            except TypeError:
                model.fit(tr['X'], tr['y'])
        except Exception:
            model.fit(tr['X'], tr['y'])

    def _score_model(self, model, va):
        predictions = model.predict(va['X'])
        if self.mode == 'class':
            return accuracy_score(va['y'].to_numpy(), predictions)
        return rmse(va['y'].to_numpy(), predictions)

    def train_models(self):
        """
        Train all configured models on the provided folds.
        """
        for model_name, template_model in self.models.items():
            print(f"Training {model_name}...")
            models = []
            scores = []
            for tr, va in tqdm(zip(self.train_splits[0], self.valid_splits[0])):
                model = copy.deepcopy(template_model)
                self._fit_model(model, tr, va)
                models.append(model)
                scores.append(self._score_model(model, va))

            self.scores[model_name] = float(np.mean(scores))
            self.trained_models[model_name] = models
            metric_name = 'accuracy' if self.mode == 'class' else 'RMSE'
            print(f"{model_name} trained with {metric_name}: {self.scores[model_name]}")
