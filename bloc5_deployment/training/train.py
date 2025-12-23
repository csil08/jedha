import pandas as pd
import numpy as np
import argparse
import os
import joblib
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor


# Definition of functions

    
def binary_to_int(X):
    """
    Converts boolean column into integers (0/1) in a dataframe
    """
    return X.astype(bool).astype(int)
    
    
def clean_df(df):
    """
    Removes useless columns and outliers
    """
    
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        
    # Outliers removal for engine_power
    q1_engine_power = df["engine_power"].quantile(0.01)
    q99_engine_power = df["engine_power"].quantile(0.99)
    df = df[(df["engine_power"] >= q1_engine_power) & (df["engine_power"] <= q99_engine_power)]

    # Outliers removal for mileage
    mask_mileage = (df['mileage']<0) | (df['mileage']>500000)
    df = df.loc[~mask_mileage]
        
    return df


def group_rare_categories(df, cols, min_freq=0.01):
    """
    Regroups rare categories (freq <= 1% by default) into "other" category
    """
    df = df.copy()
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        rares = freq[freq < min_freq].index
        df[col] = df[col].replace(rares, "other")
    return df

def rare_category_transform(X):
    """ 
    Applies group_rare_categories function on a list of columns
    """
    return group_rare_categories(
        X,
        cols=["paint_color", "fuel", "model_key"],
        min_freq=0.01
    )
        

if __name__ == "__main__":

    # Define directories
    DATA_DIR = "/home/app/data"
    MODELS_DIR = "/home/app/models"

    # Set your variables for your environment
    EXPERIMENT_NAME="get_around_pricing_model"

    # Set tracking URI to your Hugging Face application
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Set experiment's info 
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    print("Training model")
    
    # Time execution
    start_time = time.time()
    
    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False)

    # Parse arguments given in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb",
                        choices=["xgb", "rf", "ridge", "lasso"],
                        help="Model to train: XGBoost, Random forest, Ridge or Lasso")
    args = parser.parse_args()
    model_name = args.model.lower()
    
    # -----------------------------------------------
    # Import dataset, clean and perform train/test split
    # -----------------------------------------------
    #df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv")
    df = pd.read_csv(f"{DATA_DIR}/get_around_pricing_project.csv")
    
    # Remove useless columns and outliers
    df = clean_df(df)
    
    # X, y split
    target = "rental_price_per_day"
    X = df.drop(columns=[target])
    y = df[target]

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    # Define numeric and categorical features
    numeric_features = ['mileage', 'engine_power']
    categorical_features = ['model_key', 'fuel', 'paint_color',
        'car_type', 'private_parking_available', 'has_gps',
        'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
        'has_speed_regulator', 'winter_tires']
    
    # Define binary features
    binary_cols = ['private_parking_available', 'has_gps', 
                'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
                'has_speed_regulator', 'winter_tires']
    
    # -----------------------------------------------
    # Preprocessing 
    # -----------------------------------------------
    
    rare_transformer = FunctionTransformer(
        func=rare_category_transform,
        validate=False
    )
    
    binary_to_int_transformer = FunctionTransformer(
        binary_to_int,
        validate=False
    )

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    
    categorical_transformer = Pipeline([
        ("rare", rare_transformer),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", binary_to_int_transformer, binary_cols),
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )  
    
    # -----------------------------------------------
    # Models to train
    # -----------------------------------------------
    models_dict = {
        "xgb": XGBRegressor(objective='reg:squarederror', 
                            random_state=0,
                            n_estimators=300, 
                            learning_rate=0.05, 
                            max_depth=6,
                            min_child_weight=3,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=3,
                            reg_lambda=3),
        "rf": RandomForestRegressor(n_estimators=200, 
                                    max_depth=None, 
                                    min_samples_split=2,
                                    min_samples_leaf=3, 
                                    max_features='sqrt', 
                                    random_state=0),
        "ridge": Ridge(alpha=10.0, random_state=0),
        "lasso": Lasso(alpha=0.01, random_state=0)
    }

    model = models_dict[args.model]
    
    # -----------------------------------------------
    # Pipeline
    # -----------------------------------------------
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    
    # -----------------------------------------------
    # MLflow run
    # -----------------------------------------------
    
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    with mlflow.start_run(experiment_id = experiment.experiment_id) as run:
        
        pipeline.fit(X_train, y_train)
        
        predictions = pipeline.predict(X_train)
        signature = infer_signature(X_train, predictions)

        # Save the model locally
        model_path = os.path.join(MODELS_DIR, f"model_{model_name}.pkl")
        joblib.dump(pipeline, model_path)
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="get_around_pricing_model",
            signature=signature
        )

        # Test set metrics
        y_test_pred = pipeline.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        mlflow.log_metric("test_root_mean_squared_error", rmse_test)
        mlflow.log_metric("test_mean_absolute_error", mae_test)
        mlflow.log_metric("test_r2_score", r2_test)
        
    print(f"Run completed for {args.model}")
    print(f"Total training time: {time.time()-start_time}")