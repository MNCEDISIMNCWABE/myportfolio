import requests
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import pygsheets
from google.oauth2.service_account import Credentials
from oauth2client.service_account import ServiceAccountCredentials
import logging

# Define constants
slack_credentials = 'https://hooks.slack.com/services/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'
google_service_account_file = Credentials.from_service_account_file('bright-arc-328707-b5e2d782b48b.json')
file = 'bright-arc-328707-b5e2d782b48b.json'
google_sheet_id = '1wu6wT8GwPitzhY8Yov3FspFBTd2dC6znKKNcbwP70UQ'
google_sheet_name = 'share_price_forecast'
bq_project_id = 'bright-arc-328707'
bq_dataset_id = 'test'
bq_table_id = 'stock_price_predictions'
start_date = '2014-01-02'
ticker_list =  ['CLS.JO', 'GLN.JO', 'PPH.JO','WHL.JO','APN.JO','RBP.JO','PIK.JO','PPE.JO',
                'HIL.JO','SOL.JO','EXX.JO','MCG.JO','AIL.JO','TGA.JO','SSW.JO','INL.JO']
title = (f":rotating_light: Stock Price Predictions Run:")

# Set up logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

def post_to_slack(message):
    data = {
        "icon_emoji": ":white_check_mark:",
        "attachments": [{
            "color": "#9733EE",
            "fields": [{"title": title, "value": message, "short": "false"}]
        }]
    }
    requests.post(slack_credentials, json=data, headers={'Content-Type': 'application/json'}, verify=True)

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

def train_and_forecast(group):
    m = Prophet(interval_width=0.95)
    m.fit(group)
    future = m.make_future_dataframe(periods=465)
    forecast = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast['ticker'] = group['ticker'].iloc[0]
    return forecast[['ds', 'ticker', 'yhat', 'yhat_upper', 'yhat_lower']]


def preprocess_ticker_names(for_loop_forecast):
    ticker_name_mapping = {
        'CLS.JO': 'Clicks',
        'GLN.JO': 'Glencore',
        'PPH.JO': 'Pepkorh',
        'WHL.JO': 'Woolies',
        'APN.JO': 'Aspen',
        'RBP.JO': 'Royal-Bafokeng',
        'PIK.JO': 'PnP',
        'HIL.JO': 'HomeChoice',
        'SOL.JO': 'Sasol',
        'EXX.JO': 'Exxaro',
        'INL.JO': 'Investec',
        'MCG.JO': 'MultiChoice',
        'AIL.JO': 'African-Rainbow',
        'TGA.JO': 'Thungela',
        'PPE.JO': 'Purple-Group',
        'SSW.JO': 'Sibanye-Stillwater',
    }

    for_loop_forecast['ticker_name'] = for_loop_forecast['ticker'].map(ticker_name_mapping)

    return for_loop_forecast


def build_pipeline():
    start_time = datetime.now()
    try:
        df = download_data(ticker_list, start_date)
        groups_by_ticker = df.groupby('ticker')

        for_loop_forecast = pd.DataFrame()
        for ticker in ticker_list:
            group = groups_by_ticker.get_group(ticker)
            forecast = train_and_forecast(group)
            for_loop_forecast = pd.concat((for_loop_forecast, forecast))

        preprocess_ticker_names(for_loop_forecast)

        # Write to Google Sheets
        gc = pygsheets.authorize(service_file=file)
        sh = gc.open_by_key(google_sheet_id)
        try:
            wks = sh.worksheet_by_title(google_sheet_name)
        except pygsheets.exceptions.WorksheetNotFound:
            wks = sh.add_worksheet(google_sheet_name)
        wks.clear('A1', None, '*')
        wks.set_dataframe(for_loop_forecast, (1, 1), encoding='utf-8', fit=True)
        wks.frozen_rows = 1

        # Write to BigQuery
        for_loop_forecast.to_gbq(destination_table=f'{bq_project_id}.{bq_dataset_id}.{bq_table_id}',
                                 project_id=bq_project_id,
                                 credentials=google_service_account_file,
                                 chunksize=10000,
                                 progress_bar=True,
                                 if_exists='replace')

        post_to_slack(f"""
            :white_check_mark: Successful!
            company: {list(for_loop_forecast['ticker_name'].unique())}
            n_companies: {for_loop_forecast['ticker_name'].nunique()}
            training_prediction_time: {round((datetime.now() - start_time).total_seconds(), 3)} sec
        """)

    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg)

if __name__ == '__main__':
    build_pipeline()
