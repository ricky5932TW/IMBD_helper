import numpy as np
import pandas as pd

class KFoldPredictor:
    def __init__(self, kfold_splits, models, types_of_features=None):
        """
        Initialize the KFoldPredictor with k-fold splits and trained models and scaler.
        
        :param kfold_splits: Dictionary containing training and validation splits.
        :param models: Dictionary of trained models.
        """
        self.kfold_splits = kfold_splits
        self.models = models
        self.rmses = {}
        self.types_of_features = types_of_features if types_of_features is not None else [1] * len(kfold_splits['train_splits'][0][0]['X'].columns)
        self.weights = []

    def __drop_outliers(self, df, threshold=3):
        """
        Drop outliers from the DataFrame based on z-score.
        Rows with any feature exceeding the threshold are removed.
        
        :param df: DataFrame to process.
        :param threshold: Z-score threshold to identify outliers.
        :return: DataFrame with outliers removed.
        """
        z_scores = np.abs((df - df.mean()) / df.std())
        return df[(z_scores < threshold).all(axis=1)]
    
    def __calculate_weights(self):
        """
        Calculate weights for each model based on their RMSE scores.
        """
        self.weights = 1 / np.array(list(self.rmses.values()))
        self.weights /= self.weights.sum()
        print(f"Weights calculated: {self.weights}")
        


    def fit_holdout(self):
        holdout = self.kfold_splits['holdout_split'][0]
        predictions = {}
        scalers = [self.kfold_splits['train_splits'][0][i]['scaler'] for i in range(len(self.kfold_splits['train_splits'][0]))]
        gaussians = [self.kfold_splits['train_splits'][0][i]['gaussian'] for i in range(len(self.kfold_splits['train_splits'][0]))]
        encoders = [self.kfold_splits['train_splits'][0][i]['encoder'] for i in range(len(self.kfold_splits['train_splits'][0]))]

        for model_name, models in self.models.items():
            predictions_model = []
            for model, robust_scaler, gaussian, encoder in zip(models, scalers, gaussians, encoders):
                holdout_X = holdout['X'].copy()

                # Apply transformations for numerical features
                numerical_indices = [i for i, feature_type in enumerate(self.types_of_features) if feature_type == 1]
                if numerical_indices:
                    holdout_X.iloc[:, numerical_indices] = robust_scaler.transform(holdout_X.iloc[:, numerical_indices])
                    holdout_X.iloc[:, numerical_indices] = gaussian.transform(holdout_X.iloc[:, numerical_indices])

                # Apply transformations for categorical features
                categorical_indices = [i for i, feature_type in enumerate(self.types_of_features) if feature_type == 0]
                if categorical_indices:
                    holdout_X.iloc[:, categorical_indices] = encoder.transform(holdout_X.iloc[:, categorical_indices])

                # Predict
                preds = model.predict(holdout_X)
                predictions_model.append(preds)

            predictions_model = np.array(predictions_model)
            predictions[model_name] = np.mean(predictions_model, axis=0)

        for model_name, preds in predictions.items():
            rmse_value = np.sqrt(np.mean((holdout['y'].to_numpy() - preds) ** 2))
            self.rmses[model_name] = rmse_value
            print(f"RMSE for {model_name}: {rmse_value}")

        self.__calculate_weights()

        prediction_list = list(predictions.values()) 
        final_prediction = np.average(prediction_list, axis=0, weights=self.weights)
        # rmse
        final_rmse = np.sqrt(np.mean((holdout['y'].to_numpy() - final_prediction) ** 2))
        print(f"Final RMSE: {final_rmse}")
        






