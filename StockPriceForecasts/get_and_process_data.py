import pandas as pd
import yfinance as yf

def download_data(ticker_list, start_date):
    # Download and preprocess data
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]
    data.columns = data.columns.droplevel()
    data = data / 100
    data = data.reset_index()
    df = pd.melt(data, id_vars='Date', value_vars=ticker_list)
    df.columns = ['ds', 'ticker', 'y']
    return df
