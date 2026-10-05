import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor


def create_model(filename: str) -> CatBoostRegressor:
    # Load the data
    df = pd.read_csv(filename)

    # Fill missing weight values with the mean and remove rows
    # that have no market_index.
    df.loc[df["weight"] < 0, "weight"] = pd.NA
    df["weight"] = df["weight"].fillna(df["weight"].mean())

    # Extract day, month, and year from the date column.
    dates = pd.to_datetime(df["date"])
    df["day"] = dates.dt.day
    df["month"] = dates.dt.month
    df["year"] = dates.dt.year

    # Prepare features and target.
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
            "year",
        ]
    ]
    y = df["posted_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_features = [
        i for i, value in enumerate(X.columns)
        if X.iloc[:, i].dtype == "object"
    ]

    model = CatBoostRegressor(
        iterations=90,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    model = create_model("data/train-test.csv")
    model.save_model("catboost_model.cbm")