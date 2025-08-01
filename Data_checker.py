import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, TargetEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from xgboost import XGBRegressor, XGBClassifier
from compare_preprocess import PreprocessComparison
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from colorama import Fore, Style, init
init(autoreset=True)
from sklearn.preprocessing import OrdinalEncoder


class DataChecker:
    """
    A class to check and preprocess data for machine learning tasks.
    It verifies data types, applies transformations, and checks for diversity in folds.
    """
    def __init__(self, X, y, X_test, mode='reg',typeofFeatures=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.X = X
        self.y = y
        self.mode = mode
        self.typeofFeatures = typeofFeatures 
        self.X_test = X_test
        self.target_encoding = None
        self.gaussian_transformer = None
        self.robust_scaler = None
        self.kfold = None
        self.auc_score = None
        self.fold_diversity = None
        self.cat_data = None
        self.num_data = None

        self.checkDtype()

    def checkDtype(self):
        if self.X is None or self.y is None:
            raise ValueError("Input data X and y cannot be None.")
        if self.X.isnull().values.any() or self.y.isnull().values.any():
            raise ValueError("Input data contains null values. Please handle them before proceeding.")
        if len(self.X) != len(self.y):
            raise ValueError("Input data X and y must have the same number of samples.")
        if not isinstance(self.X, pd.DataFrame) or not isinstance(self.y, pd.Series):
            raise TypeError("Input data X must be a pandas DataFrame and y must be a pandas Series.")
        if self.mode not in ['reg', 'class']:
            raise ValueError("Mode must be either 'reg' for regression or 'class' for classification.")
        
    def varify_data_types(self):
        """
        [0,0, ...1,0], 0 is categorical, 1 is numerical
        """
        if self.typeofFeatures is None:
            raise ValueError("typeofFeatures must be provided to verify data types.")
        
        if len(self.typeofFeatures) != len(self.X.columns):
            raise ValueError("Length of typeofFeatures must match the number of columns in X.")
        
        self.cat_data = [col for col, typ in zip(self.X.columns, self.typeofFeatures) if typ == 0]
        self.num_data = [col for col, typ in zip(self.X.columns, self.typeofFeatures) if typ == 1]
        
        print(f"Categorical features: {self.cat_data}")
        print(f"Numerical features: {self.num_data}")
        return self.cat_data, self.num_data
    
    def apply_transformations(self):
        """
        Apply Gaussian transformation, robust scaling, and target encoding separately for categorical and numerical features.
        """
        if self.cat_data is None or self.num_data is None:
            raise ValueError("Categorical and numerical features must be identified before applying transformations.")
        
        # Apply transformations only if there are numerical features
        if len(self.num_data) > 0:
            # Robust scaling for numerical features
            self.robust_scaler = RobustScaler()
            self.X[self.num_data] = self.robust_scaler.fit_transform(self.X[self.num_data])
            self.X_test[self.num_data] = self.robust_scaler.transform(self.X_test[self.num_data])

            # Gaussian transformation for numerical features
            self.gaussian_transformer = QuantileTransformer(output_distribution='normal')
            self.X[self.num_data] = self.gaussian_transformer.fit_transform(self.X[self.num_data])
            self.X_test[self.num_data] = self.gaussian_transformer.transform(self.X_test[self.num_data])
        else:
            print("No numerical features found, skipping numerical transformations.")
        
        # Apply target encoding only if there are categorical features
        if len(self.cat_data) > 0:
            try:
                self.target_encoding = TargetEncoder()
                self.X[self.cat_data] = self.target_encoding.fit_transform(self.X[self.cat_data], self.y)
                self.X_test[self.cat_data] = self.target_encoding.transform(self.X_test[self.cat_data])
            except Exception as e:
                print(f"Error in target encoding: {e} or not enough data for target encoding.")
        else:
            print("No categorical features found, skipping target encoding.")

    def get_folds(self, n_splits=5, n_repeats=10):

        def get_repeated_kfold_splits_with_holdout(X, y, n_splits=5, n_repeats=10, holdout_size=0.1, random_state=42):
            """
            Get Repeated KFold splits with true holdout set for regression problem
            """
            # First split: separate holdout set
            X_train_cv, X_holdout, y_train_cv, y_holdout = train_test_split(
                X, y, test_size=holdout_size, random_state=random_state, stratify=None
            )
            
            # Create RepeatedKFold on the remaining data
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            
            train_splits = []
            valid_splits = []
            
            for train_idx, valid_idx in rkf.split(X_train_cv):
                train_split = {
                    'X': X_train_cv.iloc[train_idx],
                    'y': y_train_cv.iloc[train_idx] if isinstance(y_train_cv, pd.DataFrame) else y_train_cv.iloc[train_idx]
                }
                valid_split = {
                    'X': X_train_cv.iloc[valid_idx],
                    'y': y_train_cv.iloc[valid_idx] if isinstance(y_train_cv, pd.DataFrame) else y_train_cv.iloc[valid_idx]
                }
                train_splits.append(train_split)
                valid_splits.append(valid_split)
            
            # Create holdout split
            holdout_split = {
                'X': X_holdout,
                'y': y_holdout
            }
            
            return train_splits, valid_splits, holdout_split
        
        # Create Repeated KFold splits with holdout for each target variable
        train_splits_of_splitters = []
        valid_splits_of_splitters = []
        holdout_splits = []


        train_splits, valid_splits, holdout_split = get_repeated_kfold_splits_with_holdout(
            self.X, self.y, n_splits=n_splits, n_repeats=n_repeats, holdout_size=0.1, random_state=42
        )
        train_splits_of_splitters.append(train_splits)
        valid_splits_of_splitters.append(valid_splits)
        holdout_splits.append(holdout_split)

        print("Repeated KFold splits with holdout created successfully!")
        print(f"Number of CV splits per target: {len(train_splits_of_splitters[0])}")
        print(f"Training set size for first fold: {len(train_splits_of_splitters[0][0]['X'])}")
        print(f"Validation set size for first fold: {len(valid_splits_of_splitters[0][0]['X'])}")
        print(f"Holdout set size: {len(holdout_splits[0]['X'])}")
        print(f"Total original data size: {len(self.X)}")

        self.kfold = {
            'train_splits': train_splits_of_splitters,
            'valid_splits': valid_splits_of_splitters,
            'holdout_split': holdout_splits
        }
        return self.kfold
    
    def compare_train_and_test(self):
        """
        Compare training and test data using PreprocessComparison.
        """
        if self.X_test is None:
            raise ValueError("X_test must be provided for comparison.")
        
        comparison = PreprocessComparison(self.X, self.X_test, self.y)
        comparison.start_comparison()
        
        print("Comparison between training and test data completed successfully.")

    def __necessary_ensemble(self,scores):
        """
        Check if ensemble is necessary based on the scores.
        If the average score is below a certain threshold, return True.
        """
        average_score = np.mean(scores)
        threshold = 0.8  # Example threshold, adjust as needed
        std = np.std(scores)
        if std/ average_score > 0.05:
            return True

    def check_diversity(self):
        """
        Check diversity of folds.
        """
        # use xgboost to check diversity to check if the folds are diverse enough
        if self.kfold is None:
            raise ValueError("KFold splits have not been created. Please run get_folds() first.")
        scores = []
        for fold in tqdm(self.kfold['train_splits'][0]):
            X_train = fold['X']
            y_train = fold['y']
            if self.mode == 'reg':
                model = XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.2, random_state=42)
            else:
                model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.2, random_state=42)
            
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            scores.append(score)
        
        print(Fore.YELLOW+rf'Average diversity score across folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
        if self.__necessary_ensemble(scores):
            print(Fore.YELLOW+"Ensemble is necessary based on the diversity of folds.")
        else:
            print(Fore.YELLOW+"Ensemble is not necessary based on the diversity of folds.")
        print(f"Diversity scores for folds: {scores}")

if __name__ == '__main__':
    train_data_path = rf'C:\Users\E4-159\Documents\GitHub\IMBD_helper\demo_datasets\playground-series-s5e8\train.csv'  
    test_data_path = rf'C:\Users\E4-159\Documents\GitHub\IMBD_helper\demo_datasets\playground-series-s5e8\test.csv'
    data = pd.read_csv(train_data_path)
    
    # Check for missing values before fillna
    print("Quantity of NaN in data before fillna:", data.isnull().sum().sum())
    
    # filling missing values with 0
    data.fillna(0, inplace=True)

    X = data.drop(columns=['id', 'y'])
    target = data['y']
    # col iloc 1,2,3,4,6,7,8,10,12,13,14,15
    X_cat = X.iloc[:, [1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15]]
    X_num = X.iloc[:, [0, 5, 9, 11]]

    # ordinal encoding for categorical features
    X_cat = X_cat.apply(lambda col: OrdinalEncoder().fit_transform(col.values.reshape(-1, 1)).flatten() if col.dtype == 'object' else col)
    X = pd.concat([X_num, X_cat], axis=1)
    print("Data after encoding:", X.head())

    X_test = pd.read_csv(test_data_path)
    # Remove 'id' column from test data to match training data structure
    X_test = X_test.drop(columns=['id'])
    X_test_cat = X_test.iloc[:, [1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15]]
    X_test_num = X_test.iloc[:, [0, 5, 9, 11]]
    X_test_cat = X_test_cat.apply(lambda col: OrdinalEncoder().fit_transform(col.values.reshape(-1, 1)).flatten() if col.dtype == 'object' else col)
    X_test = pd.concat([X_test_num, X_test_cat], axis=1)


    
    typeofFeatures_new = [1, 1, 1, 1] + [0] * 12

    data_checker = DataChecker(X=X, y=target, mode='class', typeofFeatures=typeofFeatures_new, X_test=X_test)
    data_checker.varify_data_types()
    data_checker.apply_transformations()
    kfold_splits = data_checker.get_folds(n_splits=5, n_repeats=10)
    print("KFold splits created successfully.")
    data_checker.compare_train_and_test()
    print("Data comparison completed successfully.")
    data_checker.check_diversity()
    print("Diversity check completed successfully.")