import argparse
import json
import subprocess
import zipfile
from pathlib import Path


pd = None
np = None
DataChecker = None
FeatureSelector = None
Serch_hyperParams = None
Training_models = None
KFoldPredictor = None
OrdinalEncoder = None
XGBClassifier = None
accuracy_score = None
train_test_split = None


COMPETITION = "spaceship-titanic"
DATA_URL = "https://www.kaggle.com/competitions/spaceship-titanic/data"
REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")
TARGET = "Transported"
SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def import_dependencies():
    global pd, np, DataChecker, FeatureSelector, Serch_hyperParams
    global Training_models, KFoldPredictor, OrdinalEncoder, XGBClassifier
    global accuracy_score, train_test_split

    try:
        import numpy as np_module
        import pandas as pd_module
        from sklearn.metrics import accuracy_score as accuracy_score_func
        from sklearn.model_selection import train_test_split as train_test_split_func
        from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoderClass
        from xgboost import XGBClassifier as XGBClassifierClass

        from Data_checker import DataChecker as DataCheckerClass
        from Feature_selector import FeatureSelector as FeatureSelectorClass
        from Kfold_predictor import KFoldPredictor as KFoldPredictorClass
        from Search_hyper_params import Serch_hyperParams as Serch_hyperParamsClass
        from train_models import Training_models as TrainingModelsClass
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependencies. Install them with:\n"
            "  python -m pip install -r requirements.txt\n\n"
            f"Original import error: {exc}"
        ) from exc

    pd = pd_module
    np = np_module
    DataChecker = DataCheckerClass
    FeatureSelector = FeatureSelectorClass
    Serch_hyperParams = Serch_hyperParamsClass
    Training_models = TrainingModelsClass
    KFoldPredictor = KFoldPredictorClass
    OrdinalEncoder = OrdinalEncoderClass
    XGBClassifier = XGBClassifierClass
    accuracy_score = accuracy_score_func
    train_test_split = train_test_split_func


def dataset_ready(data_dir):
    return all((data_dir / filename).exists() for filename in REQUIRED_FILES)


def ensure_dataset(data_dir, allow_download=True):
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_ready(data_dir):
        return

    download_error = None
    if allow_download:
        try:
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(data_dir)],
                check=True,
                text=True,
            )
            zip_path = data_dir / f"{COMPETITION}.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as archive:
                    archive.extractall(data_dir)
        except (FileNotFoundError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
            download_error = exc

    if dataset_ready(data_dir):
        return

    message = (
        f"Spaceship Titanic files are missing from {data_dir}.\n"
        f"Manual setup: download train.csv, test.csv, and sample_submission.csv from {DATA_URL} "
        f"and place them in {data_dir}.\n"
        f"Kaggle API setup: kaggle competitions download -c {COMPETITION} -p {data_dir}, "
        f"then unzip {COMPETITION}.zip into that same folder."
    )
    if download_error is not None:
        message += f"\nKaggle API attempt failed with: {download_error}"
    raise FileNotFoundError(message)


def split_series(df, column, sep, width, fill_value="Missing"):
    parts = df[column].fillna(fill_value).astype(str).str.split(sep, expand=True)
    return parts.reindex(columns=range(width), fill_value=fill_value)


def engineer_features(df):
    engineered = df.copy()

    passenger_parts = split_series(engineered, "PassengerId", "_", 2, "0")
    engineered["PassengerGroup"] = pd.to_numeric(passenger_parts[0], errors="coerce")
    engineered["PassengerNumber"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    cabin_parts = split_series(engineered, "Cabin", "/", 3)
    engineered["CabinDeck"] = cabin_parts[0].replace("nan", "Missing")
    engineered["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    engineered["CabinSide"] = cabin_parts[2].replace("nan", "Missing")

    engineered["LastName"] = (
        engineered["Name"]
        .fillna("Unknown Unknown")
        .astype(str)
        .str.split()
        .str[-1]
        .fillna("Unknown")
    )

    for column in SPEND_COLUMNS:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce")
    engineered["TotalSpend"] = engineered[SPEND_COLUMNS].fillna(0).sum(axis=1)
    engineered["NoSpend"] = (engineered["TotalSpend"] == 0).astype(int)
    engineered["SpendPerAge"] = engineered["TotalSpend"] / (pd.to_numeric(engineered["Age"], errors="coerce").fillna(0) + 1)

    return engineered.drop(columns=["PassengerId", "Cabin", "Name"], errors="ignore")


def prepare_spaceship_data(train_df, test_df):
    y = train_df[TARGET].astype(int)
    X = engineer_features(train_df.drop(columns=[TARGET]))
    X_test = engineer_features(test_df)

    original_columns = X.columns.tolist()
    X_test = X_test.reindex(columns=original_columns)
    categorical_columns = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
    ]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    if numeric_columns:
        X.loc[:, numeric_columns] = X.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
        X_test.loc[:, numeric_columns] = X_test.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")

        numeric_medians = X.loc[:, numeric_columns].median(numeric_only=True).fillna(0)
        X.loc[:, numeric_columns] = X.loc[:, numeric_columns].fillna(numeric_medians)
        X_test.loc[:, numeric_columns] = X_test.loc[:, numeric_columns].fillna(numeric_medians)

    if categorical_columns:
        X.loc[:, categorical_columns] = X.loc[:, categorical_columns].fillna("Missing").astype(str)
        X_test.loc[:, categorical_columns] = X_test.loc[:, categorical_columns].fillna("Missing").astype(str)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_train = encoder.fit_transform(X.loc[:, categorical_columns])
        encoded_test = encoder.transform(X_test.loc[:, categorical_columns])

        encoded_train_df = pd.DataFrame(encoded_train, columns=categorical_columns, index=X.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=categorical_columns, index=X_test.index)
        X = X.drop(columns=categorical_columns).join(encoded_train_df).reindex(columns=original_columns)
        X_test = X_test.drop(columns=categorical_columns).join(encoded_test_df).reindex(columns=original_columns)

    # Final numeric coercion keeps preprocessing resilient across pandas dtype backends.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    type_of_features = [0 if column in categorical_columns else 1 for column in X.columns]
    return X, y, X_test, type_of_features, categorical_columns, numeric_columns


def write_submission(path, passenger_ids, predictions):
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        TARGET: predictions.astype(bool),
    })
    submission.to_csv(path, index=False)


def run_default_xgb(X, y, X_test, passenger_ids, output_dir, random_state, holdout_size):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y,
    )
    holdout_model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    holdout_model.fit(X_train, y_train)
    holdout_predictions = holdout_model.predict(X_valid)
    holdout_accuracy = float(accuracy_score(y_valid, holdout_predictions))

    final_model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test)

    submission_path = output_dir / "submission_default_xgb.csv"
    write_submission(submission_path, passenger_ids, test_predictions)

    return {
        "submission": str(submission_path),
        "holdout_accuracy": holdout_accuracy,
    }


def parse_thresholds(value):
    if value is None:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def select_features(X, y, X_test, type_of_features, args):
    feature_candidates = parse_thresholds(args.feature_thresholds)
    selector = FeatureSelector(
        X=X,
        y=y,
        mode="class",
        TypesofFeatures=feature_candidates,
        use_gpu=args.use_gpu,
        cv=args.feature_cv,
        min_selected_features=args.min_selected_features,
    )
    baseline_scores = selector.get_baseline()
    threshold_scores = selector.get_scores_with_different_thresholds()
    X_selected, fitted_selector = selector.get_new_dataset()
    X_test_selected = pd.DataFrame(
        fitted_selector.transform(X_test),
        columns=X_selected.columns,
        index=X_test.index,
    )

    type_map = dict(zip(X.columns, type_of_features))
    selected_types = [type_map[column] for column in X_selected.columns]
    return X_selected, X_test_selected, selected_types, {
        "method": "xgb_rfe",
        "baseline_cv_accuracy": float(np.mean(baseline_scores)),
        "best_feature_count": int(selector.best_feature_count),
        "threshold_cv_accuracy": float(np.max([np.mean(scores) for scores in threshold_scores])),
        "selected_feature_count": int(X_selected.shape[1]),
        "selected_features": X_selected.columns.tolist(),
    }


def default_ensemble_params():
    return {
        "xgboost": {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        "randomforest": {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        },
        "catboost": {
            "iterations": 500,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 5.0,
        },
        "extratrees": {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        },
        "lgbm": {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
    }


def search_params(folds, args):
    params = default_ensemble_params()
    if args.optuna_trials <= 0:
        if args.use_gpu:
            params["xgboost"]["device"] = "cuda"
        return params

    searcher = Serch_hyperParams(
        train=folds["train_splits"],
        val=folds["valid_splits"],
        mode="class",
        use_gpu=args.use_gpu,
    )
    params["xgboost"] = searcher.search_in_xgboost(args.optuna_trials)
    params["randomforest"] = searcher.search_in_randomforest(args.optuna_trials)
    params["catboost"] = searcher.search_in_catboost(args.optuna_trials)
    params["extratrees"] = searcher.search_in_extratrees(args.optuna_trials)
    params["lgbm"] = searcher.search_in_lgbm(args.optuna_trials)
    if args.use_gpu:
        params["xgboost"]["device"] = "cuda"
    return params


def run_custom_ensemble(X, y, X_test, type_of_features, passenger_ids, output_dir, args):
    X_selected, X_test_selected, selected_types, selection_metrics = select_features(
        X,
        y,
        X_test,
        type_of_features,
        args,
    )

    data_checker = DataChecker(
        X=X_selected,
        y=y,
        X_test=X_test_selected,
        mode="class",
        typeofFeatures=selected_types,
    )
    data_checker.varify_data_types()
    folds = data_checker.get_folds(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        control_scaler=False,
        control_gaussian=False,
        control_targetencoder=False,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
    )

    params = search_params(folds, args)
    trainer = Training_models(
        train=folds["train_splits"],
        val=folds["valid_splits"],
        params=params,
        mode="class",
    )
    trainer.train_models()

    predictor = KFoldPredictor(
        kfold_splits=folds,
        models=trainer.trained_models,
        types_of_features=selected_types,
        mode="class",
    )
    final_holdout_accuracy = predictor.fit_holdout()
    test_predictions = predictor.predict(X_test_selected)

    submission_path = output_dir / "submission_custom_ensemble.csv"
    write_submission(submission_path, passenger_ids, test_predictions)

    return {
        "submission": str(submission_path),
        "holdout_accuracy": final_holdout_accuracy,
        "selection": selection_metrics,
        "cv_accuracy_by_model": trainer.scores,
        "holdout_accuracy_by_model": predictor.metrics,
        "ensemble_weights": [float(value) for value in predictor.weights],
        "model_params": params,
    }


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_submission(path, expected_ids):
    submission = pd.read_csv(path)
    expected_columns = ["PassengerId", TARGET]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"{path} must have columns {expected_columns}, got {submission.columns.tolist()}.")
    if len(submission) != len(expected_ids):
        raise ValueError(f"{path} has {len(submission)} rows; expected {len(expected_ids)}.")
    if not submission["PassengerId"].equals(expected_ids.reset_index(drop=True)):
        raise ValueError(f"{path} PassengerId order does not match test.csv.")
    if submission[TARGET].dtype != bool:
        raise ValueError(f"{path} Transported column must be boolean.")


def run(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(data_dir, allow_download=not args.no_download)
    import_dependencies()

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    passenger_ids = test_df["PassengerId"].copy()

    X, y, X_test, type_of_features, categorical_columns, numeric_columns = prepare_spaceship_data(train_df, test_df)

    metrics = {
        "dataset": {
            "source": DATA_URL,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "features_after_engineering": int(X.shape[1]),
            "categorical_features": categorical_columns,
            "numeric_features": numeric_columns,
        },
        "default_xgb": run_default_xgb(
            X,
            y,
            X_test,
            passenger_ids,
            output_dir,
            args.random_state,
            args.holdout_size,
        ),
        "custom_ensemble": run_custom_ensemble(
            X,
            y,
            X_test,
            type_of_features,
            passenger_ids,
            output_dir,
            args,
        ),
    }

    for filename in ["submission_default_xgb.csv", "submission_custom_ensemble.csv"]:
        validate_submission(output_dir / filename, passenger_ids)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2), encoding="utf-8")
    print(f"Default XGB submission: {metrics['default_xgb']['submission']}")
    print(f"Custom ensemble submission: {metrics['custom_ensemble']['submission']}")
    print(f"Metrics: {metrics_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run Spaceship Titanic default XGB vs custom ensemble benchmark.")
    parser.add_argument("--data-dir", default="demo_datasets/spaceship-titanic")
    parser.add_argument("--output-dir", default="outputs/spaceship-titanic")
    parser.add_argument("--no-download", action="store_true", help="Do not try the Kaggle API when data files are missing.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-size", type=float, default=0.1)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--feature-cv", type=int, default=10)
    parser.add_argument("--feature-candidates", dest="feature_thresholds", default="", help="RFE candidate list using integer feature counts only (e.g. 1,2,3,4). Leave empty to try all counts 1..n_features.")
    parser.add_argument("--feature-thresholds", dest="feature_thresholds", help="Deprecated alias of --feature-candidates")
    parser.add_argument("--min-selected-features", type=int, default=1, help="Minimum number of features to keep when selecting the best RFE candidate.")
    parser.add_argument("--optuna-trials", type=int, default=100, help="Trials per ensemble model. Use 0 for fixed defaults.")
    parser.add_argument("--use-gpu", action="store_true", help="Pass GPU hints to XGBoost feature selection/search.")
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
