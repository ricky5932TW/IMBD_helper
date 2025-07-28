import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Matplotlib or seaborn not installed. Skipping plotting functionality.")

class FeatureSelector:
    def __init__(self, X, y, mode, 
                 TypesofFeatures=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TypesofFeatures = TypesofFeatures
        self.X = X
        self.y = y
        self.mode = mode # 'reg' or 'class'
        self.model = None
        self.base_line_scores = None
        self.thres_scores = None

        self.checkDtype()

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
        
    def get_baseline(self):
        """
        Calculate the baseline score using cross-validation.
        """
        if self.mode == 'reg':
            self.model = xgb.XGBRegressor()
        else:
            self.model = xgb.XGBClassifier()

        scores = cross_val_score(self.model, self.X, self.y, cv=10, scoring='neg_mean_squared_error' if self.mode == 'reg' else 'accuracy')
        self.base_line_scores = list(np.abs(scores))
        return list(np.abs(scores))
    
    def get_scores_with_different_thresholds(self):
        """
        Get scores for different feature selection thresholds.
        """
        # Ensure model is initialized
        if self.model is None:
            if self.mode == 'reg':
                self.model = xgb.XGBRegressor()
            else:
                self.model = xgb.XGBClassifier()

        scores = []
        for threshold in tqdm(self.TypesofFeatures):
            selector = SelectFromModel(self.model, threshold=f"{threshold}*median")
            X_selected = selector.fit_transform(self.X, self.y)
            
            # Check if any features were selected
            if X_selected.shape[1] == 0:
                raise ValueError(f"No features selected for threshold {threshold}. Please adjust the threshold or check your data.")
            
            score = cross_val_score(self.model, X_selected, self.y, cv=10, scoring='neg_mean_squared_error' if self.mode == 'reg' else 'accuracy')
            scores.append(list(np.abs(score)))
        self.thres_scores = scores
        return scores
    
    def draw_scores_diagram(self, percentages=[0.5, 0.6, 0.9]):
        """
        Draw a diagram with mean maximum and minimum scores for different thresholds. and vertical lines for specified percentages scores.
        """
        if self.thres_scores is None:
            raise ValueError("Threshold scores have not been calculated. Please run get_scores_with_different_thresholds() first.")
        
        means = [np.mean(score) for score in self.thres_scores]
        maxs = [np.max(score) for score in self.thres_scores]
        mins = [np.min(score) for score in self.thres_scores]
        max_idx = np.argmax(means)
        print(f"Maximum score at threshold {self.TypesofFeatures[max_idx]}: {means[max_idx]}")

        plt.figure(figsize=(10, 6))
        plt.plot(self.TypesofFeatures, means, label='Mean Score', marker='o')
        plt.fill_between(self.TypesofFeatures, mins, maxs, alpha=0.2, label='Min-Max Range')
        
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
        plt.axhline(y=max_score, color='g', linestyle='--', label='Baseline Score')
        plt.xlabel('Feature Selection Thresholds')
        plt.ylabel('Scores')
        plt.title('Feature Selection Scores with Different Thresholds')
        plt.legend()
        plt.grid() 
        plt.show()

if __name__ == '__main__':
    # Example usage
    # Assuming 'data' is a pandas DataFrame and 'target' is a pandas Series
    train_data_path = rf'C:\Users\E4-159\Documents\GitHub\IMBD_helper\demo_datasets\playground-series-s5e3\train.csv'  # Replace with your actual data path
    # drop id column if exists
    data = pd.read_csv(train_data_path)
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    # filling missing values with 0
    data.fillna(0, inplace=True)
    X = data.drop(columns=['rainfall'])  # Assuming 'target' is the column to predict
    target = data['rainfall']  # Assuming 'target' is the column to predict

    feature_selector = FeatureSelector(X=X, y=target, mode='class')
    
    baseline_scores = feature_selector.get_baseline()
    print("Baseline scores:", np.mean(baseline_scores))

    scores_with_thresholds = feature_selector.get_scores_with_different_thresholds()


    feature_selector.draw_scores_diagram()
    print("Diagram drawn successfully.")


        

    