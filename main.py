from Data_checker import DataChecker
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from Feature_selector import FeatureSelector
from L1_model_zoo import *
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # 限制最多使用 4 個核心
from Search_hyper_params import Serch_hyperParams


if __name__ == '__main__':
    all_train_data = pd.read_csv(rf'C:\Users\E4-159\Documents\py_surr\imbd2025\初\final_dataset_with_statics.csv')
    # drop Disp. X and Disp. Z columns for y
    X = all_train_data.drop(columns=['Time','Disp. X', 'Disp. Z', '日期'])
    y_x = all_train_data['Disp. X']
    y_z = all_train_data['Disp. Z']
    '''pick Categorical features: ['轉速 (rpm)', '轉速 (rpm).1', '溫度']
    Numerical features: ['Spindle Motor', 'X Motor', 'Z Motor', 'PT01_mean', 'PT01_skew', 'PT02_std', 'PT03_std', 'PT03_skew', 'PT06_std', 'PT07_skew', 'PT07_kurtosis', 'PT09_skew', 'PT11_skew', 'TC02_std', 'TC02_skew', 'TC03_skew', 'TC04_skew', 'TC06_kurtosis', 'TC07_median', 'TC07_quantile_75%', 'TC07_kurtosis', 'X Motor_skew', 'X Motor_kurtosis', 'Z Motor_skew']'''
    X = X[['轉速 (rpm)', '轉速 (rpm).1', '溫度','Spindle Motor', 'X Motor', 'Z Motor', 'PT01_mean', 'PT01_skew', 'PT02_std', 'PT03_std', 'PT03_skew', 'PT06_std', 'PT07_skew', 'PT07_kurtosis', 'PT09_skew', 'PT11_skew', 'TC02_std', 'TC02_skew', 'TC03_skew', 'TC04_skew', 'TC06_kurtosis', 'TC07_median', 'TC07_quantile_75%', 'TC07_kurtosis', 'X Motor_skew', 'X Motor_kurtosis', 'Z Motor_skew']]
    # x size
    print("X shape:", X.shape)
    print("Number of features:", X.shape[1])
    
    # Create typeofFeatures to match the number of features
    # Assuming first 25 are numerical and remaining are categorical
    #typeofFeatures_new = [1] * (X.shape[1] - 11) + [0] * 11
    typeofFeatures_new = [0] * 3 + [1] * (X.shape[1] - 3)  # First 3 are categorical, rest are numerical
    
    # Create a test dataset with the same structure as X (DataFrame) instead of numpy array
    X_test_dummy = pd.DataFrame(np.zeros((X.shape[0], X.shape[1])), columns=X.columns)
    
    data_checker = DataChecker(X=X, y=y_z, mode='reg', typeofFeatures=typeofFeatures_new, X_test=X_test_dummy)
    data_checker.varify_data_types()
    data_checker.apply_transformations(use_target_encoder=False)  # Use False to skip Target Encoding for now
    kfold_splits = data_checker.get_folds(n_splits=5, n_repeats=10)    

    do = Serch_hyperParams(train=kfold_splits['train_splits'], val=kfold_splits['valid_splits'], mode='reg', use_gpu=True)
    #xgboost_params = do.search_in_xgboost()   #done
    #randomforest_params = do.search_in_randomforest()
    #catboost_params = do.search_in_catboost()
    #extratrees_params = do.search_in_extratrees()
    lgbm_params = do.search_in_lgbm()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    print("KFold splits created successfully.")
    #data_checker.compare_train_and_test()
    print("Data comparison completed successfully.")
    #data_checker.check_diversity()
    print("Diversity check completed successfully.")

    # Use y0 (first target column) for feature selection instead of undefined 'target'
    feature_selector = FeatureSelector(X=X, y=y_z, mode='reg', use_gpu=True)
    
    _ = feature_selector.get_baseline()
    _ = feature_selector.get_scores_with_different_thresholds()


    feature_selector.draw_scores_diagram(save=True)
    print("Diagram drawn successfully.")

    new_dataset, selector = feature_selector.get_new_dataset()
    #selected features

    X_test_dummy = pd.DataFrame(np.zeros((new_dataset.shape[0], new_dataset.shape[1])), columns=new_dataset.columns)
    feature_types = {
        # 類別型
        '轉速 (rpm)': 0, '進給 (mm/min)': 0, '時間 (Hr)': 0,
        '轉速 (rpm).1': 0, '進給 (mm/min).1': 0, '時間 (Hr).1': 0,
        '轉速 (rpm).2': 0, '進給 (mm/min).2': 0, '時間 (Hr).2': 0,
        '控溫': 0, '溫度': 0
    }
    # typeofFeatures 直接用 new_dataset.columns
    # if not in feature_types, default to numerical
    # 1 for numerical, 0 for categorical
    typeofFeatures_new = [feature_types.get(col, 1) for col in new_dataset.columns]
    selected_data_checker = DataChecker(X=new_dataset, y=y_z, mode='reg', typeofFeatures=typeofFeatures_new, X_test=X_test_dummy)
    selected_data_checker.varify_data_types()
    selected_data_checker.apply_transformations(use_target_encoder=False)  # Use False to skip Target
    splited_data = selected_data_checker.get_folds(n_splits=5, n_repeats=10)
    # save folds to a pickle file
    with open('train_splits.pkl', 'wb') as f:
        pickle.dump(splited_data, f)
    print("New dataset created and split successfully.")


    get_models_base_scores(splited_data['train_splits'], splited_data['valid_splits'])

    get_residual_correlation(splited_data['train_splits'], splited_data['valid_splits'], draw_diagram=True)

    # do y_z