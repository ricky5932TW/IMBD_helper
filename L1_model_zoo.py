import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
# lgbm
from lightgbm import LGBMRegressor
#catboost
from catboost import CatBoostRegressor
#MLP
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error as rmse
from tqdm import tqdm

# calculate the coefficient of residuals through the base model
def calculate_residuals(model, X, y):
    """
    Calculate the residuals of the model predictions.
    """
    predictions = model.predict(X)
    if isinstance(y, pd.DataFrame):
        # 將 (n, 1) 的 DataFrame 轉為一維陣列
        y = y.values.ravel()
    residuals = y - predictions
    return residuals

def fit_base_model(model, tr_split, va_split):
    """
    Fit the base model and return the residuals.
    """
    model.fit(tr_split['X'], tr_split['y'])
    residuals = calculate_residuals(model, va_split['X'], va_split['y'])
    return residuals

def calculate_corelation_between_scores(scores):
    """
    Calculate the correlation between the scores.
    """
    scores_df = pd.DataFrame(scores)
    correlation_matrix = scores_df.corr()
    return correlation_matrix

def get_models_base_scores(train_splits_of_splitters, valid_splits_of_splitters):
    model_scores = {
        'Ridge': [], 'Linear': [], 'Lasso': [], 
        'SVR': [], 'KNN': [], 'RF': [], 'XGB': [],
        'LGBM': [], 'CatBoost': [], 'MLP': [],
        'ExtraTrees': []
    }

    # Extract the actual splits from the nested structure
    train_splits = train_splits_of_splitters[0]
    valid_splits = valid_splits_of_splitters[0]

    for tr, va in tqdm(zip(train_splits, valid_splits)):
        # Initialize models
        Ridge_model = Ridge()
        Linear_model = LinearRegression()
        Lasso_model = Lasso()
        SVR_model = SVR()
        KNN_model = KNeighborsRegressor()
        RF_model = RandomForestRegressor(random_state=42)
        XGB_model = XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse')
        
        # Fit Ridge model
        Ridge_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), Ridge_model.predict(va['X']))
        model_scores['Ridge'].append(score)
        
        # Fit Linear model
        Linear_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), Linear_model.predict(va['X']))
        model_scores['Linear'].append(score)
        
        # Fit Lasso model
        Lasso_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), Lasso_model.predict(va['X']))
        model_scores['Lasso'].append(score)
        
        # Fit SVR model
        SVR_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), SVR_model.predict(va['X']))
        model_scores['SVR'].append(score)
        
        # Fit KNN model
        KNN_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), KNN_model.predict(va['X']))
        model_scores['KNN'].append(score)
        
        # Fit RF model
        RF_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), RF_model.predict(va['X']))
        model_scores['RF'].append(score)
        
        # Fit XGB model with its specific parameters
        XGB_model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])], verbose=0)
        score = rmse(va['y'].to_numpy(), XGB_model.predict(va['X']))
        model_scores['XGB'].append(score)

        # Fit LGBM model
        LGBM_model = LGBMRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42,verbose=-1)
        LGBM_model.fit(tr['X'], tr['y'], eval_set=[(va['X'], va['y'])])
        score = rmse(va['y'].to_numpy(), LGBM_model.predict(va['X']))
        model_scores['LGBM'].append(score)

        # Fit CatBoost model
        CatBoost_model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=5, random_state=42, verbose=0)
        CatBoost_model.fit(tr['X'], tr['y'], eval_set=(va['X'], va['y']))
        score = rmse(va['y'].to_numpy(), CatBoost_model.predict(va['X']))
        model_scores['CatBoost'].append(score)
        # Fit MLP model 
        MLP_model = MLPRegressor(hidden_layer_sizes=(256,), max_iter=1000, random_state=42)
        MLP_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), MLP_model.predict(va['X']))
        model_scores['MLP'].append(score)
        # Fit ExtraTrees model
        ExtraTrees_model = ExtraTreesRegressor(n_estimators=500, random_state=42)
        ExtraTrees_model.fit(tr['X'], tr['y'])
        score = rmse(va['y'].to_numpy(), ExtraTrees_model.predict(va['X']))
        model_scores['ExtraTrees'].append(score)

    # Convert scores to DataFrame for easier analysis
    scores_df = pd.DataFrame(model_scores)

    # Calculate the average score for each model
    average_scores = scores_df.mean()

    # Show the average scores
    print("Average scores of the base models:")
    print(average_scores)

    # Calculate standard deviation of scores to see model stability
    std_scores = scores_df.std()
    print("\nStandard deviation of scores:")
    print(std_scores)

    # Show the best model
    best_model = average_scores.idxmin()
    print(f"\nBest model: {best_model} with average RMSE: {average_scores[best_model]:.4f}")