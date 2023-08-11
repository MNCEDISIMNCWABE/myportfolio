import pandas as pd
from pytrends.request import TrendReq
import numpy as np
from google.oauth2 import service_account
import pygsheets
from constants import *

def get_google_trends_data(credentials_path):
    pytrends = TrendReq()
    pytrends.build_payload(kw_list, timeframe=timeframe, geo="ZA", gprop="")
    df_regions = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=True)
    df_regions = df_regions.reset_index()

    # Don't convert geo codes to int, keep them as strings
    df_regions = df_regions.astype({'geoName': str})

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

    write_to_gsheet(file, sheet_id, sheet_name, df_regions)

    df_regions = df_regions.astype(int)
    return df_regions
