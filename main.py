from Data_checker import DataChecker
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from Feature_selector import FeatureSelector
from L1_model_zoo import *

if __name__ == '__main__':
    """
    train_data_path = 'C:/Users/E4-159/Documents/GitHub/IMBD_helper/demo_datasets/playground-series-s5e8/train.csv'  
    test_data_path = 'C:/Users/E4-159/Documents/GitHub/IMBD_helper/demo_datasets/playground-series-s5e8/test.csv'
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
    """
    all_train_data = pd.read_csv('C:/Users/E4-159/Documents/py_surr/imbd2022-main/training.csv')
    all_test_data = pd.read_csv('C:/Users/E4-159/Documents/py_surr/imbd2022-main/testing.csv')
    X = all_train_data.drop(columns=['sensor_point5_i_value', 'sensor_point6_i_value','sensor_point7_i_value','sensor_point8_i_value','sensor_point9_i_value','sensor_point10_i_value'], axis=1)
    y = all_train_data[['sensor_point5_i_value', 'sensor_point6_i_value','sensor_point7_i_value','sensor_point8_i_value','sensor_point9_i_value','sensor_point10_i_value']]
    X_test = all_test_data.drop(columns=['update_time', 'create_time'], axis=1)
    y_test = pd.read_csv('C:/Users/E4-159/Documents/py_surr/imbd2022-main/testing_ans.csv')
    
    # Create Series instead of DataFrame for targets (use single brackets)
    y0 = y['sensor_point5_i_value']  # Series instead of DataFrame
    y1 = y['sensor_point6_i_value']
    y2 = y['sensor_point7_i_value'] 
    y3 = y['sensor_point8_i_value']
    y4 = y['sensor_point9_i_value']
    y5 = y['sensor_point10_i_value']

    y_test_0 = y_test['sensor_point5_i_value']  # Series instead of DataFrame

    #fillna with 0
    X.fillna(0, inplace=True)
    # fillna with 0 in test data
    X_test.fillna(0, inplace=True)
    
    # Also fill null values in target variables (now they are Series)
    y0 = y0.fillna(0)
    y1 = y1.fillna(0) 
    y2 = y2.fillna(0)
    y3 = y3.fillna(0)
    y4 = y4.fillna(0)
    y5 = y5.fillna(0)
    
    # Check for any remaining null values
    print("Null values in X:", X.isnull().sum().sum())
    print("Null values in X_test:", X_test.isnull().sum().sum())
    print("Null values in y0:", y0.isnull().sum())

    # x size
    print("X shape:", X.shape)
    
    typeofFeatures_new = [1] * 125
    data_checker = DataChecker(X=X, y=y0, mode='reg', typeofFeatures=typeofFeatures_new, X_test=X_test)
    data_checker.varify_data_types()
    data_checker.apply_transformations()
    kfold_splits = data_checker.get_folds(n_splits=5, n_repeats=10)
    print("KFold splits created successfully.")
    data_checker.compare_train_and_test()
    print("Data comparison completed successfully.")
    data_checker.check_diversity()
    print("Diversity check completed successfully.")

    # Use y0 (first target column) for feature selection instead of undefined 'target'
    feature_selector = FeatureSelector(X=X, y=y0, mode='reg', use_gpu=True)
    
    _ = feature_selector.get_baseline()
    _ = feature_selector.get_scores_with_different_thresholds()


    feature_selector.draw_scores_diagram(save=True)
    print("Diagram drawn successfully.")

    new_dataset, selector = feature_selector.get_new_dataset()
    new_dataset_test = selector.transform(data_checker.X_test)

    # turn to new dataset into Datachecker and split into train and test
    data_checker.X = new_dataset
    data_checker.X_test = new_dataset_test
    splited_data = data_checker.get_folds(n_splits=5, n_repeats=10)
    print("New dataset created and split successfully.")


    get_models_base_scores(splited_data['train_splits'], splited_data['valid_splits'])