import pandas as pd


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