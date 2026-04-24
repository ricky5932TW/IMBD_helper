import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, RepeatedStratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
from compare_preprocess import PreprocessComparison
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from colorama import Fore, Style, init
init(autoreset=True)
from sklearn.preprocessing import OrdinalEncoder

try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    TargetEncoder = None


class DataChecker:
    """
    A class to check and preprocess data for machine learning tasks.
    It verifies data types, applies transformations, and checks for diversity in folds.
    """
    def __init__(self, X, y, X_test, mode='reg',typeofFeatures=None):
        # Remove the super().__init__() call since DataChecker doesn't inherit from any specific class

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

        self.scalers = []
        self.transformers = []
        self.target_encoders = []
        self.ordinal_encoders = {}
        self.text_columns = []

        self.checkDtype()
        self.auto_encode_text_columns()

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

    def auto_encode_text_columns(self):
        """
        Auto-detect object / category columns and apply OrdinalEncoder.
        Fits on train+test combined so unseen categories in test don't break.
        Also auto-populates typeofFeatures (0=cat, 1=num) when not provided.
        """
        text_cols = [
            col for col in self.X.columns
            if self.X[col].dtype == 'object' or str(self.X[col].dtype) == 'category'
        ]

        if text_cols:
            print(Fore.CYAN + f"Auto-detected text/object columns: {text_cols}")
            for col in text_cols:
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1
                )
                train_vals = self.X[col].astype(str).values.reshape(-1, 1)
                if self.X_test is not None and col in self.X_test.columns:
                    test_vals = self.X_test[col].astype(str).values.reshape(-1, 1)
                    combined = np.concatenate([train_vals, test_vals], axis=0)
                    encoder.fit(combined)
                    self.X_test[col] = encoder.transform(test_vals).flatten()
                else:
                    encoder.fit(train_vals)
                self.X[col] = encoder.transform(train_vals).flatten()
                self.ordinal_encoders[col] = encoder
            print(Fore.GREEN + f"Encoded {len(text_cols)} text columns with OrdinalEncoder.")
        else:
            if self.typeofFeatures is not None:
                provided_cat_count = sum(typ == 0 for typ in self.typeofFeatures)
                if provided_cat_count > 0:
                    print("No raw text/object columns detected; categorical columns appear to be already encoded.")
                else:
                    print("No text/object columns detected; all features are numeric.")
            else:
                print("No text/object columns detected, skipping auto-encoding.")

        self.text_columns = text_cols

        if self.typeofFeatures is None:
            self.typeofFeatures = [
                0 if col in text_cols else 1
                for col in self.X.columns
            ]
            n_cat = sum(t == 0 for t in self.typeofFeatures)
            n_num = sum(t == 1 for t in self.typeofFeatures)
            print(Fore.CYAN + f"Auto-populated typeofFeatures: {n_cat} categorical, {n_num} numerical")

        return text_cols
        
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
    
    def apply_transformations(self, use_target_encoder=False):
        """
        Apply Gaussian transformation, robust scaling, and target encoding separately for categorical and numerical features.
        """
        if self.cat_data is None or self.num_data is None :
            raise ValueError("Categorical and numerical features must be identified before applying transformations.")
        
        # Apply transformations only if there are numerical features
        if len(self.num_data) > 0 and False:
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
        if len(self.cat_data) > 0 and use_target_encoder and False:
            try:
                self.target_encoding = TargetEncoder()
                self.X[self.cat_data] = self.target_encoding.fit_transform(self.X[self.cat_data], self.y)
                self.X_test[self.cat_data] = self.target_encoding.transform(self.X_test[self.cat_data])
            except Exception as e:
                print(f"Error in target encoding: {e} or not enough data for target encoding.")
        else:
            print("No categorical features found, skipping target encoding.")

    def get_folds(
        self,
        n_splits=5,
        n_repeats=10,
        control_scaler=True,
        control_gaussian=True,
        control_targetencoder=True,
        holdout_size=0.1,
        random_state=42,
    ):


        # 根據 typeofFeatures 分辨數值型與類別型
        if self.typeofFeatures is None:
            raise ValueError("typeofFeatures must be provided to verify data types.")
        if len(self.typeofFeatures) != len(self.X.columns):
            raise ValueError("Length of typeofFeatures must match the number of columns in X.")
        num_cols = [col for col, typ in zip(self.X.columns, self.typeofFeatures) if typ == 1]
        cat_cols = [col for col, typ in zip(self.X.columns, self.typeofFeatures) if typ == 0]

        # 用來存所有 fold/scaler/encoder
        self.folds_scalers = []
        self.folds_gaussians = []
        self.folds_encoders = []
        self.cv_scaler = None
        self.cv_gaussian = None
        self.cv_encoder = None

        def get_repeated_kfold_splits_with_holdout(X, y, n_splits=5, n_repeats=10, holdout_size=0.1, random_state=42):
            """
            Get Repeated KFold splits with true holdout set for regression/classification problem
            """
            stratify_y = y if self.mode == 'class' else None
            # First split: separate holdout set
            X_train_cv, X_holdout, y_train_cv, y_holdout = train_test_split(
                X, y, test_size=holdout_size, random_state=random_state, stratify=stratify_y
            )


           
            if self.mode == 'class':
                rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                split_iterator = rkf.split(X_train_cv, y_train_cv)
            else:
                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                split_iterator = rkf.split(X_train_cv)

            train_splits = []
            valid_splits = []


            for train_idx, valid_idx in split_iterator:
                X_train_fold = X_train_cv.iloc[train_idx].copy()
                X_valid_fold = X_train_cv.iloc[valid_idx].copy()
                y_train_fold = y_train_cv.iloc[train_idx] if isinstance(y_train_cv, pd.DataFrame) else y_train_cv.iloc[train_idx]
                y_valid_fold = y_train_cv.iloc[valid_idx] if isinstance(y_train_cv, pd.DataFrame) else y_train_cv.iloc[valid_idx]

                scaler_fold = None
                gaussian_fold = None
                encoder_fold = None
                # fold 內也做 RobustScaler + Gaussian Rank
                if len(num_cols) > 0:
                    if control_scaler:
                        scaler_fold = RobustScaler()
                        X_train_fold[num_cols] = scaler_fold.fit_transform(X_train_fold[num_cols])
                        X_valid_fold[num_cols] = scaler_fold.transform(X_valid_fold[num_cols])
                    if control_gaussian:
                        gaussian_fold = QuantileTransformer(output_distribution='normal')
                        X_train_fold[num_cols] = gaussian_fold.fit_transform(X_train_fold[num_cols])
                        X_valid_fold[num_cols] = gaussian_fold.transform(X_valid_fold[num_cols])
                if len(cat_cols) > 0 and control_targetencoder:
                    try:
                        if TargetEncoder is None:
                            raise ImportError("TargetEncoder is not available in this scikit-learn version.")
                        encoder_fold = TargetEncoder()
                        X_train_fold[cat_cols] = encoder_fold.fit_transform(X_train_fold[cat_cols], y_train_fold)
                        X_valid_fold[cat_cols] = encoder_fold.transform(X_valid_fold[cat_cols])
                        encoded_features = list(getattr(encoder_fold, 'feature_names_in_', cat_cols))
                        missing = [c for c in cat_cols if c not in encoded_features]
                        if missing:
                            raise RuntimeError(f"Target encoding missed columns: {missing}")
                    except Exception as e:
                        print(f"Error in fold target encoding: {e}")

                # 存進 self
                self.folds_scalers.append(scaler_fold)
                self.folds_gaussians.append(gaussian_fold)
                self.folds_encoders.append(encoder_fold)

                train_split = {
                    'X': X_train_fold,
                    'y': y_train_fold,
                    'scaler': scaler_fold,
                    'gaussian': gaussian_fold,
                    'encoder': encoder_fold
                }
                valid_split = {
                    'X': X_valid_fold,
                    'y': y_valid_fold,
                    'scaler': scaler_fold,
                    'gaussian': gaussian_fold,
                    'encoder': encoder_fold
                }
                train_splits.append(train_split)
                valid_splits.append(valid_split)

            # Create holdout split
            holdout_split = {
                'X': X_holdout,
                'y': y_holdout,
            }

            return train_splits, valid_splits, holdout_split

        # Create Repeated KFold splits with holdout for each target variable
        train_splits_of_splitters = []
        valid_splits_of_splitters = []
        holdout_splits = []

        train_splits, valid_splits, holdout_split = get_repeated_kfold_splits_with_holdout(
            self.X.copy(), self.y.copy(), n_splits=n_splits, n_repeats=n_repeats, holdout_size=0.1, random_state=42
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

        if control_targetencoder and len(cat_cols) > 0:
            encoded_folds = sum(enc is not None for enc in self.folds_encoders)
            print(Fore.GREEN + f"Target encoding applied to {len(cat_cols)} categorical columns "
                              f"across {encoded_folds}/{len(self.folds_encoders)} folds.")
            print(Fore.GREEN + f"Encoded categorical columns: {cat_cols}")

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
                model = XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.1, random_state=42)
            else:
                model = XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.1, random_state=42)
            
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

    X_test = pd.read_csv(test_data_path)
    X_test = X_test.drop(columns=['id'])

    # DataChecker auto-detects object columns and encodes them,
    # and auto-populates typeofFeatures based on dtype when omitted.
    data_checker = DataChecker(X=X, y=target, mode='class', X_test=X_test)
    data_checker.varify_data_types()
    data_checker.apply_transformations()
    kfold_splits = data_checker.get_folds(n_splits=5, n_repeats=10)
    print("KFold splits created successfully.")
    data_checker.compare_train_and_test()
    print("Data comparison completed successfully.")
    data_checker.check_diversity()
    print("Diversity check completed successfully.")
