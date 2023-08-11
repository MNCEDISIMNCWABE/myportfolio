import pandas as pd
from prophet import Prophet

def train_and_forecast(group):
    m = Prophet(interval_width=0.95)
    m.fit(group)
    future = m.make_future_dataframe(periods=465)
    forecast = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast['ticker'] = group['ticker'].iloc[0]
    return forecast[['ds', 'ticker', 'yhat', 'yhat_upper', 'yhat_lower']]
