from Data_checker import DataChecker
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from Feature_selector import FeatureSelector
from L1_model_zoo import *
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # 限制最多使用 4 個核心


if __name__ == '__main__':

    all_train_data = pd.read_csv(rf'C:\Users\E4-159\Documents\py_surr\imbd2025\初\final_combined_data.csv')
    # drop Disp. X and Disp. Z columns for y
    X = all_train_data.drop(columns=['Time','Disp. X', 'Disp. Z','日期'])
    y_x = all_train_data['Disp. X']
    y_z = all_train_data['Disp. Z']

    X_lag1 = X.copy().shift(1)
    # replace column names to avoid confusion
    X_lag1.columns = [f"{col}_lag1" for col in X.columns]
    # drop the last row of X_lag1
    X_lag1 = X_lag1.iloc[1:].reset_index(drop=True)

    #drop the first row of X 
    X = X.iloc[1:].reset_index(drop=True)
    y_x = y_x.iloc[1:].reset_index(drop=True)
    y_z = y_z.iloc[1:].reset_index(drop=True)

    X = pd.concat([X, X_lag1], axis=1)
    # print sum of nan
    print("Sum of NaN in X after adding lag1 features:", X.isnull().sum().sum())
    # fill NaN values with 0
    X.fillna(0, inplace=True)
 
    # x size
    print("X shape:", X.shape)
    print("Number of features:", X.shape[1])

    # head
    print("X head:", X.head())
    # kill  the program
    #raise SystemExit("Stopping execution for debugging purposes.")    
    # Create typeofFeatures to match the number of features
    # Assuming first 25 are numerical and remaining are categorical
    typeofFeatures_new = [1] * 24 + [0] * (int(X.shape[1]/2) - 24) + [1] * 24 + [0] * (int(X.shape[1]/2) - 24)

    
    # Create a test dataset with the same structure as X (DataFrame) instead of numpy array
    X_test_dummy = pd.DataFrame(np.zeros((X.shape[0], X.shape[1])), columns=X.columns)
    
    data_checker = DataChecker(X=X, y=y_x, mode='reg', typeofFeatures=typeofFeatures_new, X_test=X_test_dummy)
    data_checker.varify_data_types()
    data_checker.apply_transformations(use_target_encoder=False)  # Use False to skip Target Encoding for now
    kfold_splits = data_checker.get_folds(n_splits=5, n_repeats=10)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    print("KFold splits created successfully.")
    #data_checker.compare_train_and_test()
    print("Data comparison completed successfully.")
    #data_checker.check_diversity()
    print("Diversity check completed successfully.")

    # Use y0 (first target column) for feature selection instead of undefined 'target'
    feature_selector = FeatureSelector(X=X, y=y_x, mode='reg', use_gpu=True)
    
    _ = feature_selector.get_baseline()
    _ = feature_selector.get_scores_with_different_thresholds()


    feature_selector.draw_scores_diagram(save=True)
    print("Diagram drawn successfully.")

    new_dataset, selector = feature_selector.get_new_dataset()
    #selected features

    X_test_dummy = pd.DataFrame(np.zeros((new_dataset.shape[0], new_dataset.shape[1])), columns=new_dataset.columns)
    feature_types = {
        # 數值型
        'PT01': 1, 'PT02': 1, 'PT03': 1, 'PT04': 1, 'PT05': 1, 'PT06': 1, 'PT07': 1, 'PT08': 1, 'PT09': 1, 'PT10': 1,
        'PT11': 1, 'PT12': 1, 'PT13': 1, 'TC01': 1, 'TC02': 1, 'TC03': 1, 'TC04': 1, 'TC05': 1, 'TC06': 1, 'TC07': 1, 'TC08': 1,
        'Spindle Motor': 1, 'X Motor': 1, 'Z Motor': 1,
        # 類別型
        '轉速 (rpm)': 0, '進給 (mm/min)': 0, '時間 (Hr)': 0,
        '轉速 (rpm).1': 0, '進給 (mm/min).1': 0, '時間 (Hr).1': 0,
        '轉速 (rpm).2': 0, '進給 (mm/min).2': 0, '時間 (Hr).2': 0,
        '控溫': 0, '溫度': 0
    }
    feature_types_lag = feature_types.copy()
    for col, tp in feature_types.items():
        feature_types_lag[f"{col}_lag1"] = tp

    # typeofFeatures 直接用 new_dataset.columns
    typeofFeatures_new = [feature_types_lag.get(col, 1) for col in new_dataset.columns]
    selected_data_checker = DataChecker(X=new_dataset, y=y_x, mode='reg', typeofFeatures=typeofFeatures_new, X_test=X_test_dummy)
    selected_data_checker.varify_data_types()
    selected_data_checker.apply_transformations(use_target_encoder=False)  # Use False to skip Target
    splited_data = selected_data_checker.get_folds(n_splits=5, n_repeats=10)
    # save folds to a pickle file
    with open('train_splits.pkl', 'wb') as f:
        pickle.dump(splited_data, f)
    print("New dataset created and split successfully.")


    get_models_base_scores(splited_data['train_splits'], splited_data['valid_splits'])

    get_residual_correlation(splited_data['train_splits'], splited_data['valid_splits'], draw_diagram=True)