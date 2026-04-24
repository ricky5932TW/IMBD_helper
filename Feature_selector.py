import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Matplotlib or seaborn not installed. Skipping plotting functionality.")

try:
    from torch import cuda
except ImportError:
    cuda = None


class FeatureSelector:
    def __init__(self, X, y, mode, 
                 TypesofFeatures=None,
                 use_gpu=False, cv=10, min_selected_features=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TypesofFeatures = TypesofFeatures if TypesofFeatures is not None else []
        self.X = X
        self.y = y
        self.mode = mode # 'reg' or 'class'
        self.model = None
        self.base_line_scores = None
        self.thres_scores = None
        self.threshold_feature_counts = None
        self.best_threshold = None
        self.best_feature_count = None
        self.cv = cv
        self.min_selected_features = int(max(1, min_selected_features))
        self.selector = None
        
        self.gpu_available = cuda is not None and cuda.is_available() and cuda.device_count() > 0
        self.use_gpu = use_gpu

        self.checkDtype()
        self.init_model()

    def _candidate_to_feature_count(self, candidate, total_features):
        """
        Convert candidate to absolute feature count for RFE.
        Only integer feature counts are accepted (no ratio mapping).
        """
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return None

        if value <= 0 or not np.isclose(value, round(value)):
            return None

        count = int(round(value))

        return int(np.clip(count, 1, total_features))

    def _build_feature_count_candidates(self, total_features):
        """
        Build integer feature-count candidates.
        If no valid candidate is provided, try all counts 1..total_features.
        """
        if not self.TypesofFeatures:
            return list(range(1, total_features + 1))

        candidates = []
        for candidate in self.TypesofFeatures:
            n_features = self._candidate_to_feature_count(candidate, total_features)
            if n_features is None:
                print(f"Invalid candidate {candidate}. Only integer feature counts are allowed. Skipping.")
                continue
            candidates.append(n_features)

        unique_candidates = sorted(set(candidates))
        if not unique_candidates:
            print("No valid integer feature-count candidates provided. Using full range 1..n_features.")
            return list(range(1, total_features + 1))

        return unique_candidates

    def _select_with_importance(self, threshold):
        """
        Keep importance-based selector for reference/debugging.
        This method is intentionally not used in the main pipeline.
        """
        self.init_model()
        selector = SelectFromModel(self.model, threshold=f"{threshold}*mean")
        X_selected = selector.fit_transform(self.X, self.y)
        return X_selected, selector

    def checkDtype(self):
        # not none
        if self.X is None or self.y is None:
            raise ValueError("Input data X and y cannot be None.")
        # no null values in the dataset
        if self.X.isnull().values.any() or self.y.isnull().values.any():
            raise ValueError("Input data contains null values. Please handle them before proceeding.")
        # if x and y are not of the same length
        if len(self.X) != len(self.y):
            raise ValueError("Input data X and y must have the same number of samples.")
        # if x is not a dataframe
        if not isinstance(self.X, pd.DataFrame) or not isinstance(self.y, pd.Series):
            raise TypeError("Input data X must be a pandas DataFrame and y must be a pandas Series.")
        # checking mode
        if self.mode not in ['reg', 'class']:
            raise ValueError("Mode must be either 'reg' for regression or 'class' for classification.")
        
    def init_model(self):
        if self.mode == 'reg':
            self.model = xgb.XGBRegressor(device='cuda' if self.gpu_available and self.use_gpu else 'cpu',
                                          eval_metric='rmse',)
        else:
            self.model = xgb.XGBClassifier(device='cuda' if self.gpu_available and self.use_gpu else 'cpu',
                                           eval_metric='logloss',)
        
    def get_baseline(self):
        """
        Calculate the baseline score using cross-validation.
        """
        self.init_model()
        # rmse
        scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring='neg_root_mean_squared_error' if self.mode == 'reg' else 'accuracy')
        self.base_line_scores = list(np.abs(scores)) if self.mode == 'reg' else list(np.abs(scores))
        print(f"Mean baseline score: {np.mean(self.base_line_scores)}")
        return list(np.abs(scores))
    
    def get_scores_with_different_thresholds(self):
        """
        Get CV scores for different XGB+RFE feature counts.
        """

        scores = []
        kept_candidates = []
        feature_counts = []
        total_features = self.X.shape[1]
        candidate_counts = self._build_feature_count_candidates(total_features)

        for n_features in tqdm(candidate_counts):

            self.init_model()
            selector = RFE(estimator=self.model, n_features_to_select=n_features, step=1)
            X_selected = selector.fit_transform(self.X, self.y)
            
            # Check if any features were selected
            if X_selected.shape[1] == 0:
                print(f"No features selected for candidate {n_features}. Skipping.")
                continue
            
            score = cross_val_score(self.model, X_selected, self.y, cv=self.cv, scoring='neg_root_mean_squared_error' if self.mode == 'reg' else 'accuracy')
            scores.append(list(np.abs(score)) if self.mode == 'reg' else list(np.abs(score)))
            kept_candidates.append(n_features)
            feature_counts.append(int(X_selected.shape[1]))

        if not scores:
            raise ValueError("No valid RFE candidates produced selected features. Please adjust TypesofFeatures.")

        self.TypesofFeatures = kept_candidates
        self.thres_scores = scores
        self.threshold_feature_counts = feature_counts

        mean_scores = [np.mean(score) for score in scores]
        candidate_indices = [
            idx for idx, count in enumerate(feature_counts)
            if count >= self.min_selected_features
        ]

        if not candidate_indices:
            print(
                f"No threshold kept at least {self.min_selected_features} features. "
                "Falling back to the best score among all thresholds."
            )
            candidate_indices = list(range(len(self.TypesofFeatures)))

        if self.mode == 'class':
            best_idx = max(candidate_indices, key=lambda idx: (mean_scores[idx], feature_counts[idx]))
        else:
            best_idx = min(candidate_indices, key=lambda idx: (mean_scores[idx], -feature_counts[idx]))

        self.best_feature_count = int(feature_counts[best_idx])
        # Keep legacy attribute for backwards compatibility.
        self.best_threshold = self.best_feature_count

        print("RFE summary (feature_count, selected_features, mean_cv_score):")
        for candidate, count, score in zip(self.TypesofFeatures, feature_counts, mean_scores):
            print(f"  {candidate}: {count} features, {score:.6f}")
        print(
            f"Best RFE feature count selected: {self.best_feature_count} "
            f"(min_selected_features={self.min_selected_features})"
        )
        return scores
    
    def draw_scores_diagram(self, percentages=[0.5, 0.6, 0.9],save=False):
        """
        Draw a diagram with mean maximum and minimum scores for different thresholds. and vertical lines for specified percentages scores.
        """
        if self.thres_scores is None:
            raise ValueError("Threshold scores have not been calculated. Please run get_scores_with_different_thresholds() first.")
        
        if self.base_line_scores is None:
            raise ValueError("Baseline scores have not been calculated. Please run get_baseline() first.")
        
        means = [np.mean(score) for score in self.thres_scores]
        maxs = [np.max(score) for score in self.thres_scores]
        mins = [np.min(score) for score in self.thres_scores]
        x_axis = self.threshold_feature_counts if self.threshold_feature_counts is not None else list(range(1, len(means) + 1))
        if self.mode == 'class':
            max_idx = np.argmax(means)
        else:
            max_idx = np.argmin(means)
        print(f"Maximum score at selected feature count {x_axis[max_idx]}: {means[max_idx]}")

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, means, label='Mean Score', marker='o', color='blue')
        plt.fill_between(x_axis, mins, maxs, alpha=0.2, label='Min-Max Range', color='orange')
        
        max_score = np.min(means)
        # draw vertical lines for each  saturation percentage
        potion_map = {0.5: (1+0.5) * max_score, 
                      0.6: (1+0.4) * max_score, 
                      0.9: (1+0.1) * max_score}
        # draw vertical lines in nearest point
        # TODO: Uncomment the following lines if you want to draw vertical lines for specific percentages
        '''for percentage in percentages:
            nearest_index = np.argmin(np.abs(np.array(self.TypesofFeatures) - percentage))
            plt.axvline(x=self.TypesofFeatures[nearest_index], color='r', linestyle='--', label=f'{percentage*100}% Threshold')'''
        # plot horizontal lines for baseline scores
        plt.axhline(y=float(np.mean(self.base_line_scores)), color='g', linestyle='--', label='Baseline Score')
        plt.xlabel('Number of Selected Features (RFE)')
        plt.ylabel('Scores')
        if self.mode == 'reg':
            plt.title('Feature Selection Scores for Regression(smaller is better)')
        else:
            plt.title('Feature Selection Scores for Classification (higher is better)')
        plt.legend()
        plt.grid() 
        if save:
            plt.savefig('feature_selection_scores.png')
        plt.show()
        

    def get_new_dataset(self):
        """
        Get a new dataset with features selected based on the specified threshold.
        """
        if self.best_feature_count is None:
            raise ValueError("Feature-count scores have not been calculated. Please run get_scores_with_different_thresholds() first.")
        

        self.init_model()
        selector = RFE(estimator=self.model, n_features_to_select=self.best_feature_count, step=1)
        self.selector = selector
        X_selected = selector.fit_transform(self.X, self.y)
        
        # Check if any features were selected
        if X_selected.shape[1] == 0:
            raise ValueError(f"No features selected for RFE n_features_to_select={self.best_feature_count}. Please adjust candidate values or check your data.")
        # print before and after shape
        print(f"Original dataset shape: {self.X.shape}")
        print(f"New dataset shape after feature selection: {X_selected.shape}")
        #print column names of the new dataset
        selected_columns = self.X.columns[selector.get_support()].tolist()
        print(f"Input features before selection: {self.X.columns.tolist()}")
        print(f"Features kept by RFE ({self.best_feature_count}): {selected_columns}")
        return pd.DataFrame(X_selected, columns=selected_columns), selector

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
    X = data_checker.X
    print("Transformed data:", X.head())

    

    feature_selector = FeatureSelector(X=X, y=target, mode='class')
    
    _ = feature_selector.get_baseline()
    _ = feature_selector.get_scores_with_different_thresholds()


    feature_selector.draw_scores_diagram(save=True)
    print("Diagram drawn successfully.")

    new_dataset, selector = feature_selector.get_new_dataset()
    new_dataset_test = selector.transform(data_checker.X_test)


        

    
