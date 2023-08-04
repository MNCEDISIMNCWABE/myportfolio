# Send message to slack    
slack_credentials = 'https://hooks.slack.com/services/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'
title = (f":rotating_light: Stock Price Predictions Run:")

import requests
from datetime import datetime
import datetime as dt
import time
import json
import warnings
warnings.filterwarnings("ignore")

start_time = datetime.now()
 
def post_to_slack(message,slack_credentials):
    data = { "icon_emoji": ":white_check_mark:",
             "attachments": [{"color": "#9733EE","fields": [{"title": title,"value": message,"short": "false"}]}]}
    url = slack_credentials
    requests.post(url, json=data, headers={'Content-Type': 'application/json'}, verify=True)

if __name__ == '__main__':
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from prophet import Prophet
        import statsmodels.api as sm
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        import pygsheets
        from google.oauth2.service_account import Credentials
        from oauth2client.service_account import ServiceAccountCredentials
        from time import time
        import json
        import requests

        # Visualization
        import seaborn as sns

        # Multi-processing
        from multiprocessing import Pool, cpu_count
        import warnings
        warnings.filterwarnings("ignore")

        import logging
        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)


        credentials = Credentials.from_service_account_file('C:/Users/leemn/OneDrive/Documents/personal google service acount/bright-arc-328707-b5e2d782b48b.json')

        #start and end date for training
        start_date = '2014-01-02'
        end_date = pd.to_datetime("today").strftime("%Y-%m-%d")  

        # Download data
        ticker_list =  ['CLS.JO', 'GLN.JO', 'PPH.JO','WHL.JO','APN.JO','RBP.JO','PIK.JO','PPE.JO',
                          'HIL.JO','SOL.JO','EXX.JO','MCG.JO','AIL.JO','TGA.JO','SSW.JO','INL.JO']
        data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]
        data.columns = data.columns.droplevel()
        data = data/100

        # Release Date from the index
        data = data.reset_index()

        # Change data from the wide format to the long format
        df = pd.melt(data, id_vars='Date', value_vars= ['CLS.JO', 'GLN.JO', 'PPH.JO','WHL.JO','APN.JO','RBP.JO','PIK.JO','PPE.JO',
                                                        'HIL.JO','SOL.JO','EXX.JO','MCG.JO','AIL.JO','TGA.JO','SSW.JO','INL.JO'])
        df.columns = ['ds', 'ticker', 'y']

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Train Model>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Group the data by ticker
        groups_by_ticker = df.groupby('ticker')
        # Check the groups in the dataframe
        groups_by_ticker.groups.keys()

        def train_and_forecast(group):
            m = Prophet(interval_width = 0.95)
            m.fit(group)
            future = m.make_future_dataframe(periods = 465)
            forecast = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast['ticker'] = group['ticker'].iloc[0]
            return forecast[['ds', 'ticker', 'yhat', 'yhat_upper', 'yhat_lower']]


        for_loop_forecast = pd.DataFrame()
        # Loop through each ticker
        for ticker in ticker_list:
            group = groups_by_ticker.get_group(ticker)  
            forecast = train_and_forecast(group)
            for_loop_forecast = pd.concat((for_loop_forecast, forecast))
            
            
            
        # give the tickers clear names
        for_loop_forecast['ticker_name'] = np.where(for_loop_forecast['ticker']=='CLS.JO','Clicks',
                                  (np.where(for_loop_forecast['ticker']=='GLN.JO','Glencore',
                                  (np.where(for_loop_forecast['ticker']=='PPH.JO','Pepkorh',         
                                  (np.where(for_loop_forecast['ticker']=='WHL.JO','Woolies',        
                                  (np.where(for_loop_forecast['ticker']=='APN.JO','Aspen',  
                                  (np.where(for_loop_forecast['ticker']=='RBP.JO','Royal-Bafokeng',  
                                  (np.where(for_loop_forecast['ticker']=='PIK.JO','PnP',          
                                  (np.where(for_loop_forecast['ticker']=='HIL.JO','HomeChoice',
                                  (np.where(for_loop_forecast['ticker']=='SOL.JO','Sasol',
                                  (np.where(for_loop_forecast['ticker']=='EXX.JO','Exxaro',
                                  (np.where(for_loop_forecast['ticker']=='INL.JO','Investec',
                                  (np.where(for_loop_forecast['ticker']=='MCG.JO','MultiChoice',
                                  (np.where(for_loop_forecast['ticker']=='AIL.JO','African-Rainbow',
                                  (np.where(for_loop_forecast['ticker']=='TGA.JO','Thungela',
                                  (np.where(for_loop_forecast['ticker']=='PPE.JO','Purple-Group',
                                  (np.where(for_loop_forecast['ticker']=='SSW.JO','Sibanye-Stillwater',
                                  for_loop_forecast.ticker)))))))))))))))))))))))))))))))
    
    
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Push Predictions to Google sheet file >>>>>>>>>>>>>>>>>>>>>
        file = 'C:/Users/leemn/OneDrive/Documents/personal google service acount/bright-arc-328707-b5e2d782b48b.json'
        id = '1wu6wT8GwPitzhY8Yov3FspFBTd2dC6znKKNcbwP70UQ'
        s_n = 'share_price_forecast'


        def write_to_gsheet(service_file_path, spreadsheet_id, sheet_name, data_df):
            gc = pygsheets.authorize(service_file=service_file_path)
            sh = gc.open_by_key(spreadsheet_id)
            try:
                sh.add_worksheet(sheet_name)
            except:
                pass
            wks_write = sh.worksheet_by_title(sheet_name)
            wks_write.clear('A1',None,'*')
            wks_write.set_dataframe(data_df, (1,1), encoding='utf-8', fit=True)
            wks_write.frozen_rows = 1

        write_to_gsheet(file, id, s_n, for_loop_forecast)



        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Push predictions to BigQuery >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for_loop_forecast.to_gbq(destination_table='bright-arc-328707.test.stock_price_predictions',
                                 project_id='bright-arc-328707',
                                 credentials=credentials,
                                 chunksize=10000,
                                 progress_bar=True,
                                 if_exists='replace')
        
         # post message to slack
        post_to_slack(f"""
                      :white_check_mark: Successful!
                       company: {list(for_loop_forecast['ticker_name'].unique())}
                       n_companies: {for_loop_forecast['ticker_name'].nunique()}
                       training_prediction_time: {round((datetime.now() - start_time).total_seconds(),3)} sec
                      """
                      ,slack_credentials)
        
    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg, slack_credentials)