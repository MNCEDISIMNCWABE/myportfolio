from datetime import datetime
import pygsheets
import pandas as pd
from google.oauth2.service_account import Credentials
import logging
from post_to_slack import post_to_slack
from get_and_process_data import download_data
from forecasting import train_and_forecast
from ticker_mapping import preprocess_ticker_names
from constants import *

# Set up logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

def build_pipeline():
    start_time = datetime.now()
    for_loop_forecast = pd.DataFrame()  # Ensure initialization outside the try-except block

    for_loop_forecast = pd.DataFrame()  # Ensure initialization outside the try-except block

    try:
        df = download_data(ticker_list, start_date)
        groups_by_ticker = df.groupby('ticker')

        for ticker in ticker_list:
            try:
                group = groups_by_ticker.get_group(ticker)
                if group['y'].isna().sum() >= len(group) - 1: 
                    raise ValueError(f"Not enough data to model for {ticker}.")
                forecast = train_and_forecast(group)
                for_loop_forecast = pd.concat([for_loop_forecast, forecast])
            except Exception as e:
                post_to_slack(f"Error with ticker {ticker}: {str(e)}")
                continue
            try:
                group = groups_by_ticker.get_group(ticker)
                if group['y'].isna().sum() >= len(group) - 1:  # Check if there are sufficient non-NaN rows
                    raise ValueError(f"Not enough data to model for {ticker}.")
                forecast = train_and_forecast(group)
                for_loop_forecast = pd.concat([for_loop_forecast, forecast])
            except Exception as e:
                post_to_slack(f"Error with ticker {ticker}: {str(e)}")
                continue

        if for_loop_forecast.empty:
            raise ValueError("No forecasts were generated; all tickers failed.")

        if for_loop_forecast.empty:
            raise ValueError("No forecasts were generated; all tickers failed.")

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
