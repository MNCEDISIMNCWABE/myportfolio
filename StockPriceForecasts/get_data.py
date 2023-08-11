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

start_date = '2014-01-02'
ticker_list =  ['CLS.JO', 'GLN.JO', 'PPH.JO','WHL.JO','APN.JO','RBP.JO','PIK.JO','PPE.JO',
                'HIL.JO','SOL.JO','EXX.JO','MCG.JO','AIL.JO','TGA.JO','SSW.JO','INL.JO']

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



