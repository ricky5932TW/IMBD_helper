from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score , mean_squared_error, mean_absolute_error
# linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

import numpy as np, pandas as pd
from colorama import Fore, Style, init
init(autoreset=True)

class PreprocessComparison:
    def __init__(self, X, X_test, y,random_state=42):
        self.random_state = random_state
        self.X = X
        self.X_pesudo_labels = np.zeros((X.shape[0]))  # Placeholder for pseudo-labels
        self.X_test = X_test
        self.X_test_pesudo_labels = np.ones((X_test.shape[0]))
        self.y = y
        self.scalers = {
            'quantile': QuantileTransformer(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        self.adversarial_classifier = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_estimators=200
        )
        '''
        LogisticRegression(
            penalty='elasticnet',
            C=1.0e-3,
            random_state=random_state,
            max_iter=10000,
            solver='saga',
            class_weight='balanced',
            l1_ratio=0.5         
        )'''

        self.iwcv_regression = XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state
        )
 

        self.scaler_scores = {
            'quantile': [],
            'standard': [],
            'minmax': [],
            'robust': [],
            'raw': []  # Placeholder for raw data scores
        }

        self.iwcv_scores = []

    def do_adv_auc_compare(self):
        """
        Compare the adversarial AUC scores for different scalers, including raw data.
        
        Returns:
        scores: List of AUC scores for each fold.
        """
        scores = []
        # Add raw data as a "scaler"
        self.scalers['raw'] = None

        for scaler_name, scaler in self.scalers.items():
            print(f"Using {scaler_name} scaler...")

            if scaler_name == 'raw':
                # Use raw data without scaling
                X_scaled = self.X
                X_test_scaled = self.X_test
            else:
                # Fit the scaler on the training data
                X_scaled = scaler.fit_transform(self.X)
                X_test_scaled = scaler.transform(self.X_test)

            # Combine the training and test data for adversarial training
            X_combined = np.vstack((X_scaled, X_test_scaled))
            y_combined = np.hstack((self.X_pesudo_labels, self.X_test_pesudo_labels))

            # Create a stratified K-Fold cross-validator
            skf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            auc_scores = []
            for train_index, test_index in skf.split(X_combined, y_combined):
                X_train, X_test = X_combined[train_index], X_combined[test_index]
                y_train, y_test = y_combined[train_index], y_combined[test_index]

                # Train the adversarial classifier
                self.adversarial_classifier.fit(X_train, y_train)

                # Predict probabilities for the test set
                y_pred_proba = self.adversarial_classifier.predict_proba(X_test)[:, 1]

                # Calculate AUC score
                auc_score = roc_auc_score(y_test, y_pred_proba)
                auc_scores.append(auc_score)
            
                # Store the scores
                self.scaler_scores[scaler_name].append(auc_score)
            print(f"{scaler_name} AUC scores: {auc_scores}")
            scores.extend(auc_scores)

        return scores
    
    def do_improved_weighted_cv(self):
        """
        Perform improved weighted cross-validation using the regression model.
        
        Returns:
        scores: List of RMSE scores for each fold.
        """
        '''
        scores = []
        #group_kfold = GroupKFold(n_splits=5)
        gkf = GroupKFold(n_splits=5)
        for scaler_name, scaler in self.scalers.items():
            print(f"Using {scaler_name} scaler for improved weighted CV...")
            # Fit the scaler on the training data
            X_scaled = scaler.fit_transform(self.X)
            X_test_scaled = scaler.transform(self.X_test)

            # Combine the training and test data for adversarial training
            X_combined = np.vstack((X_scaled, X_test_scaled))
            y_combined = np.hstack((self.X_pesudo_labels, self.X_test_pesudo_labels))

            # Create a GroupKFold cross-validator
            gkf = GroupKFold(n_splits=5)
            for train_index, test_index in gkf.split(X_combined, y_combined, groups=y_combined):
                X_train, X_test = X_combined[train_index], X_combined[test_index]
                y_train, y_test = y_combined[train_index], y_combined[test_index]

                # Train the regression model
                self.iwcv_regression.fit(X_train, y_train, sample_weight=np.ones_like(y_train))

                # Predict on the test set
                y_pred = self.iwcv_regression.predict(X_test)

                # Calculate RMSE score
                rmse_score = mean_squared_error(y_test, y_pred, squared=False)
                scores.append(rmse_score)
            
                print(f"{scaler_name} RMSE score: {rmse_score:.4f}")
        '''

    
    def show_results(self):
        """
        Print the AUC scores for each scaler.
        """
        print(Fore.YELLOW+"\nAdversarial AUC Scores:")
        for scaler_name, scores in self.scaler_scores.items():
            print(f"{scaler_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


    def start_comparison(self):
        """
        Start the comparison of different preprocessing methods.
        """
        print("Starting adversarial AUC comparison...")
        scores = self.do_adv_auc_compare()
        self.show_results()
        return scores
    

if __name__ == "__main__":
    all_train_data = pd.read_csv(rf'.\data\train.csv',dtype=np.float32)
    all_test_data = pd.read_csv(rf'.\data\test.csv',dtype=np.float32)
    X = all_train_data.drop(columns=['SeqNo', 'O'], axis=1)
    y = all_train_data['O']
    X_test = all_test_data.drop(columns=['SeqNo', 'O'], axis=1)
    y_test = all_test_data['O']

    preprocess_comparison = PreprocessComparison(X, X_test, y)
    _ = preprocess_comparison.start_comparison()

        
    



