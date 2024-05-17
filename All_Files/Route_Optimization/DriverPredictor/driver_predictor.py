#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Send message to slack    
slack_credentials = 'https://hooks.slack.com/services/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'
title = (f":rotating_light: Orders and Drivers Predictions Run:")

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
        import pandas_gbq
        from prophet import Prophet
        import statsmodels.api as sm
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        from google.oauth2.service_account import Credentials
        #from oauth2client.service_account import ServiceAccountCredentials
        from tqdm import tqdm
        import time
        #import pygsheets
        import datetime as dt
        from datetime import datetime
        import time
        # Multi-processing
        from multiprocessing import Pool, cpu_count
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)


        credentials = Credentials.from_service_account_file('C:/Users/leemn/OneDrive/Documents/Service Account/cb-prod-297913-d207de781afd.json')
        
        #>>>>>>>>>>>>>>>>>>> Pull Training Data for Nando's SA >>>>>>>>>>>>>>>>
        query_drivers = '''
                      SELECT * FROM `cb-prod-297913.driver_predictor.training_data` 
                      WHERE client_name IN ("Nando's SA")
                      '''
        df_drivers = pd.read_gbq(query_drivers, project_id='cb-prod-297913', 
                                 dialect='standard', 
                                 #progress_bar_type='tqdm_notebook',
                                 credentials=credentials)


        # Orders & Drivers data
        data_orders = df_drivers[['ds','branch_name_received','number_of_orders']]
        data_orders.columns = ['ds','branch_name_received', 'y']
        data_drivers = df_drivers[['ds','branch_name_received','number_of_drivers']]
        data_drivers.columns = ['ds','branch_name_received', 'y']


        ##>>>>>>>>>>>>>>>> STAGE 1: Hourly Order Prediction
        df_grouped_orders = data_orders.groupby('branch_name_received').filter(lambda x: len(x) >= 2)
        final_forecast_orders = pd.DataFrame(columns=['branch_name_received','ds','yhat'])
        grouped_orders = df_grouped_orders.groupby('branch_name_received')
        for branch in grouped_orders.groups:
            group_orders = grouped_orders.get_group(branch)
            m_orders = Prophet(interval_width=0.95)
            m_orders.fit(group_orders)
            future_orders = m_orders.make_future_dataframe(periods=168, freq='H')
            forecast_orders = m_orders.predict(future_orders)
            forecast_orders['branch_name_received'] = branch
            final_forecast_orders = pd.concat([final_forecast_orders, forecast_orders], ignore_index=True)



        ##>>>>>>>>>>>>>>>>> STAGE 2: Hourly Driver Prediction
        df_grouped_drivers = data_drivers.groupby('branch_name_received').filter(lambda x: len(x) >= 2)
        final_forecast_drivers = pd.DataFrame(columns=['branch_name_received','ds','yhat'])
        grouped_drivers = df_grouped_drivers.groupby('branch_name_received')
        for branch in grouped_drivers.groups:
            group_drivers = grouped_drivers.get_group(branch)
            m_drivers = Prophet(interval_width=0.95)
            m_drivers.fit(group_drivers)
            future_drivers = m_drivers.make_future_dataframe(periods=168, freq='H')
            forecast_drivers = m_drivers.predict(future_drivers)
            forecast_drivers['branch_name_received'] = branch
            final_forecast_drivers = pd.concat([final_forecast_drivers, forecast_drivers], ignore_index=True)

        ## Combined Predictions
        final_preds = pd.merge(final_forecast_orders[['ds','branch_name_received','yhat']].rename(columns={'ds':'Hour','yhat':'predicted_orders'}) 
                       ,final_forecast_drivers[['ds','branch_name_received','yhat']].rename(columns={'ds':'Hour','yhat':'predicted_drivers'})
                       ,on=['Hour','branch_name_received'])

        final_preds['Hour'] = final_preds['Hour'].astype('datetime64[ns]')
        final_preds['predicted_orders'] = final_preds['predicted_orders'].astype('float64')
        final_preds['predicted_drivers'] = final_preds['predicted_drivers'].astype('float64')

        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Push predictions to BigQuery >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        final_preds.to_gbq(destination_table = 'cb-prod-297913.driver_predictor.cb-predicted_drivers',
               project_id = 'cb-prod-297913',
               credentials = credentials,
               chunksize = 50000,
               #progress_bar = True,
               if_exists = 'replace')
        
        # post message to slack
        post_to_slack(f"""
                      :white_check_mark: Successful!
                       client: {df_drivers['client_name'].unique()}
                       n_branches: {final_preds['branch_name_received'].nunique()}
                       training_prediction_time: {round((datetime.now() - start_time).total_seconds(),3)} sec
                      """
                      ,slack_credentials)
        
    except:
        error_msg = f":warning: Oops, Error in the script!!!"
        post_to_slack(error_msg,slack_credentials)


# In[ ]:




