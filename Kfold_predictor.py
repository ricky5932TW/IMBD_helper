import numpy as np
import pandas as pd


class KFoldPredictor:
    def __init__(self, kfold_splits, models, types_of_features=None, mode='reg'):
        """
        Evaluate fold-trained models on a holdout set and predict new rows.
        """
        self.kfold_splits = kfold_splits
        self.models = models
        self.mode = mode
        self.metrics = {}
        self.rmses = self.metrics
        self.types_of_features = types_of_features if types_of_features is not None else [1] * len(kfold_splits['train_splits'][0][0]['X'].columns)
        self.weights = []

    def __calculate_weights(self):
        values = np.array(list(self.metrics.values()), dtype=float)
        if self.mode == 'class':
            weights = np.maximum(values, 1e-6)
        else:
            weights = 1 / np.maximum(values, 1e-6)
        self.weights = weights / weights.sum()
        print(f"Weights calculated: {self.weights}")

    def _transform_with_fold_artifacts(self, X, robust_scaler, gaussian, encoder):
        transformed = X.copy()
        original_columns = transformed.columns.tolist()

        numerical_indices = [i for i, feature_type in enumerate(self.types_of_features) if feature_type == 1]
        if numerical_indices:
            numerical_columns = transformed.columns[numerical_indices].tolist()
            numerical_frame = transformed.loc[:, numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            if robust_scaler is not None:
                numerical_frame = pd.DataFrame(
                    robust_scaler.transform(numerical_frame),
                    columns=numerical_columns,
                    index=transformed.index,
                )
            if gaussian is not None:
                numerical_frame = pd.DataFrame(
                    gaussian.transform(numerical_frame),
                    columns=numerical_columns,
                    index=transformed.index,
                )
            transformed = (
                transformed.drop(columns=numerical_columns)
                .join(numerical_frame.astype(float))
                .reindex(columns=original_columns)
            )

        categorical_indices = [i for i, feature_type in enumerate(self.types_of_features) if feature_type == 0]
        if categorical_indices and encoder is not None:
            categorical_columns = transformed.columns[categorical_indices].tolist()
            categorical_frame = pd.DataFrame(
                encoder.transform(transformed.loc[:, categorical_columns]),
                columns=categorical_columns,
                index=transformed.index,
            )
            transformed = (
                transformed.drop(columns=categorical_columns)
                .join(categorical_frame.astype(float))
                .reindex(columns=original_columns)
            )

        return transformed.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    def _predict_model_output(self, model, X):
        if self.mode == 'class':
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            return model.predict(X).astype(float)
        return model.predict(X)

    def _classification_labels(self, predictions):
        predictions = np.asarray(predictions)
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        return (predictions.ravel() >= 0.5).astype(int)

    def _predict_each_model(self, X):
        predictions = {}
        scalers = [self.kfold_splits['train_splits'][0][i]['scaler'] for i in range(len(self.kfold_splits['train_splits'][0]))]
        gaussians = [self.kfold_splits['train_splits'][0][i]['gaussian'] for i in range(len(self.kfold_splits['train_splits'][0]))]
        encoders = [self.kfold_splits['train_splits'][0][i]['encoder'] for i in range(len(self.kfold_splits['train_splits'][0]))]

        for model_name, models in self.models.items():
            fold_predictions = []
            for model, robust_scaler, gaussian, encoder in zip(models, scalers, gaussians, encoders):
                fold_X = self._transform_with_fold_artifacts(X, robust_scaler, gaussian, encoder)
                fold_predictions.append(self._predict_model_output(model, fold_X))
            predictions[model_name] = np.mean(np.array(fold_predictions), axis=0)

        return predictions

    def fit_holdout(self):
        holdout = self.kfold_splits['holdout_split'][0]
        predictions = self._predict_each_model(holdout['X'])

        for model_name, preds in predictions.items():
            if self.mode == 'class':
                score = float(np.mean(self._classification_labels(preds) == holdout['y'].to_numpy()))
                self.metrics[model_name] = score
                print(f"Accuracy for {model_name}: {score}")
            else:
                score = float(np.sqrt(np.mean((holdout['y'].to_numpy() - preds) ** 2)))
                self.metrics[model_name] = score
                print(f"RMSE for {model_name}: {score}")

        self.__calculate_weights()
        final_prediction = self._weighted_prediction(predictions)

        if self.mode == 'class':
            final_score = float(np.mean(self._classification_labels(final_prediction) == holdout['y'].to_numpy()))
            print(f"Final accuracy: {final_score}")
            return final_score

        final_score = float(np.sqrt(np.mean((holdout['y'].to_numpy() - final_prediction) ** 2)))
        print(f"Final RMSE: {final_score}")
        return final_score

    def _weighted_prediction(self, predictions):
        prediction_list = list(predictions.values())
        if len(self.weights) != len(prediction_list):
            self.weights = np.ones(len(prediction_list)) / len(prediction_list)
        return np.average(prediction_list, axis=0, weights=self.weights)

    def predict(self, X, return_proba=False):
        predictions = self._predict_each_model(X)
        final_prediction = self._weighted_prediction(predictions)
        if self.mode == 'class' and not return_proba:
            return self._classification_labels(final_prediction)
        return final_prediction
