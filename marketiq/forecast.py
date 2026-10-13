import pandas as pd
from sklearn.linear_model import LinearRegression


def prophet_forecast(df_monthly: pd.DataFrame, periods: int = 6):

    try:
        from prophet import Prophet
    except Exception:
        return None, "Prophet not installed", 0

    if df_monthly.empty:
        return None, "No data", 0

    # Prepare data
    fit_df = df_monthly.rename(
        columns={
            "month": "ds",
            "revenue": "y"
        }
    )

    fit_df = fit_df[["ds", "y"]].copy()

    # Make sure dates are sorted
    fit_df = fit_df.sort_values("ds").reset_index(drop=True)

    # ==========================================
    # TRAIN / TEST SPLIT
    # ==========================================

    # Keep the last 3 months for testing
    if len(fit_df) >= 9:

        train_df = fit_df.iloc[:-3].copy()
        test_df = fit_df.iloc[-3:].copy()

    else:

        # Not enough data for proper testing
        train_df = fit_df.copy()
        test_df = pd.DataFrame()


    # ==========================================
    # TRAIN MODEL
    # ==========================================

    model = Prophet(
        seasonality_mode="multiplicative"
    )

    model.fit(train_df)


    # ==========================================
    # MODEL EVALUATION
    # ==========================================

    if not test_df.empty:

        test_prediction = model.predict(
            test_df[["ds"]]
        )

        evaluation = test_df.merge(
            test_prediction[["ds", "yhat"]],
            on="ds"
        )

        # Mean Absolute Error
        mae = (
            evaluation["y"] - evaluation["yhat"]
        ).abs().mean()

        # MAPE
        mape = (
            (
                (evaluation["y"] - evaluation["yhat"]).abs()
                / evaluation["y"]
            ) * 100
        ).mean()

        # Simple percentage representation
        accuracy = max(
            0,
            100 - mape
        )

    else:

        mae = 0
        accuracy = 0


    # ==========================================
    # FUTURE FORECAST
    # ==========================================

    # Train final model using all available historical data
    final_model = Prophet(
        seasonality_mode="multiplicative"
    )

    final_model.fit(fit_df)

    future = final_model.make_future_dataframe(
        periods=periods,
        freq="MS"
    )

    prediction = final_model.predict(
        future
    )


    # Get only future months
    forecast = prediction[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(periods)


    return forecast, mae, accuracy

def linear_regression_forecast(df_monthly: pd.DataFrame, periods: int = 6):
    if df_monthly.empty:
        return None, "No data", 0

    data = df_monthly[["month", "revenue"]].copy()
    data["month"] = pd.to_datetime(data["month"])
    data = data.sort_values("month").reset_index(drop=True)

    # Convert dates into numeric time steps
    data["time"] = range(len(data))

    # Keep the last 3 months for testing when enough data exists
    if len(data) >= 9:
        train = data.iloc[:-3]
        test = data.iloc[-3:]

        model = LinearRegression()
        model.fit(train[["time"]], train["revenue"])

        predicted = model.predict(test[["time"]])

        mae = abs(test["revenue"].to_numpy() - predicted).mean()

        actual = test["revenue"].to_numpy()
        nonzero = actual != 0
        if nonzero.any():
            mape = (
                abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero]).mean()
                * 100
            )
            accuracy = max(0, 100 - mape)
        else:
            accuracy = 0
    else:
        mae = 0
        accuracy = 0

    # Train the final model on all historical data
    final_model = LinearRegression()
    final_model.fit(data[["time"]], data["revenue"])

    future_steps = list(range(len(data), len(data) + periods))
    future_revenue = final_model.predict(
        pd.DataFrame({"time": future_steps})
    )

    last_month = data["month"].max()
    future_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": future_revenue,
    })

    return forecast, mae, accuracy