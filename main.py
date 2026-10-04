import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor

MODEL_FILENAME = "catboost_model.cbm"
TRAINING_DATA_FILENAME = "data/validation.csv"
TEMPLATE_FILENAME = "data/validation-predictions-template.csv"
RESULTS_FILENAME = "data/validation_predictions.csv"

def load_model(filename: str)-> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(filename)
    return model

def predict(model: CatBoostRegressor, filename: str)-> pd.DataFrame:
    # Load the data
    df = pd.read_csv(filename)
    df["weight"] = df["weight"].fillna(df["weight"].mean())
    dates = pd.to_datetime(df["date"])
    df["day"] = dates.dt.day
    df["month"] = dates.dt.month
    df["year"] = dates.dt.year
    # Prepare the features for prediction
    X = df[
        [
            "pickup",
            "delivery",
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "equipment",
            "weight",
            "day",
            "month",
            "year"
            ]
        ]
    predictions = model.predict(X)
    return pd.DataFrame({"load_id": df["load_id"], "predicted_rate": predictions})

if __name__ == "__main__":
    model = load_model(MODEL_FILENAME)
    predictions_df = predict(model, TRAINING_DATA_FILENAME)
    template_ids = pd.read_csv(TEMPLATE_FILENAME)["load_id"]
    predictions_df = pd.merge(pd.DataFrame({"load_id": template_ids}), predictions_df, on="load_id", how="left")
    predictions_df.to_csv(RESULTS_FILENAME, index=False)
