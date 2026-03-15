import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('train_hourly.csv')


def augment_df(df, devices, dropId = True):
    df = df.merge(devices, on="deviceId", how="left")

    df["timedate"] = pd.to_datetime(df["hour"], utc=True)

    # encode time features
    df["year"] = df["timedate"].dt.year
    df["month"] = df["timedate"].dt.month
    df["hour"] = df["timedate"].dt.hour

    df["month_cos"] = np.cos(2 * np.pi * (df["month"]) / 12)

    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + 5) / 24)

    df.drop(columns=["timedate", "hour", "period"], inplace=True)

    if dropId:
        df.drop(columns=["deviceId", "year", "month"], inplace=True) 

    temp_cols = [col for col in df.columns if col.startswith("t")]

    
    #ENGINEERED FEATURES
    df['a1'] = df['t4'] * df['t8']
    df['a2'] = df['t4'] / df['t8']
    df['a4'] = df['t4'] + df['t8']
    df['a5'] = df['t4'] * df['t13']

    df['b1'] = df['t5'] - df['t3']
    df['b2'] = df['t6'] - df['t4']

    df['b3'] = df['x3'] * (1 - df['t1_max'])

    df["c1"] = df["t3"] - df["t4"]
    df["c2"] = df["t5"] - df["t6"]
    df["c3"] = df["t10"] - df["t11"]

    return df


def train_XGB(train_df, params):
    target = "x2"

    X = train_df.drop(columns=[target])
    y = train_df[target]

    split_idx = int(len(train_df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1800,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=40,
    )

    # Predict
    preds = model.predict(dtest)

    # Evaluate
    rmse = mean_absolute_error(y_test, preds)
    print("MAE:", rmse)

    return model

def train_XGB_on_full_dataset(train_df, params):
    target = "x2"

    X = train_df.drop(columns=[target])
    y = train_df[target]

    X_train = X

    y_train = y

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Model parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 12,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train")],
        early_stopping_rounds=20,
    )

    return model

def save_submission(valid_df, test_df, model):
    target = "x2"

    full_test = pd.concat([valid_df, test_df], ignore_index=True)
    feature_cols = [c for c in full_test.columns if c not in ['x2', 'deviceId', 'year', 'month']]

    X_submit = full_test[feature_cols]
    dsubmit = xgb.DMatrix(X_submit)
    full_test['prediction'] = np.clip(model.predict(dsubmit), 0, None)

    submission = (
        full_test.groupby(['deviceId', 'year', 'month'])['prediction']
        .mean()
        .reset_index()
    )

    # 5. Format and Save
    # Ensure the columns match the submission requirements exactly
    submission = submission[['deviceId', 'year', 'month', 'prediction']]
    pd.DataFrame.to_parquet(submission, 'submission.parquet')
    submission.to_csv("submission.csv", index=False)

    print(f"Submission file created with {len(submission)} rows.")
    print(submission.head())


def main():
    devices = pd.read_csv('devices.csv')
    devices

    train_df = augment_df(pd.read_csv('train_hourly.csv'), devices)
    valid_df = augment_df(pd.read_csv('valid_hourly.csv'), devices, dropId=False)
    test_df = augment_df(pd.read_csv('test_hourly.csv'), devices, dropId=False)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 12,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    model = train_XGB_on_full_dataset(train_df, params)

    save_submission(valid_df, test_df, model)



