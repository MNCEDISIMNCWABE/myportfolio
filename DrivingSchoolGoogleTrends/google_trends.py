import pandas as pd
from pytrends.request import TrendReq
import numpy as np
from google.oauth2 import service_account
import pygsheets
from constants import kw_list, timeframe

def get_google_trends_data(credentials_path):
    #kw_list = ["driving school", "learners license", "driving license", "driving schools near me", "code 14 driving school"]
    #timeframe = "now 1-d"

    pytrends = TrendReq()
    pytrends.build_payload(kw_list, timeframe=timeframe, geo="ZA", gprop="")
    df_regions = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=True)
    df_regions = df_regions.reset_index()

    file = credentials_path
    sheet_id = '1TDg0Gb_M5LFd-zOi70nmw2dyCpACqMi5qLDJTl4xxFk'
    sheet_name = 'trends'

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
