# Send message to slack    
slack_credentials = 'https://hooks.slack.com/services/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'
title = (f":rotating_light: Driving School Search Terms - Google Trends Past 24 hours:")

import requests
 
def post_to_slack(message,slack_credentials):
    data = { "icon_emoji": ":white_check_mark:",
             "attachments": [{"color": "#9733EE","fields": [{"title": title,"value": message,"short": "false"}]}]}
    url = slack_credentials
    requests.post(url, json=data, headers={'Content-Type': 'application/json'}, verify=True)

if __name__ == '__main__':
    try:
        
        #pip install --upgrade google-auth
        import pandas as pd
        from pytrends.request import TrendReq
        import numpy as np
        #from oauth2client.service_account import ServiceAccountCredentials
        from google.oauth2 import service_account
        from google.oauth2.service_account import Credentials
        import pygsheets
        import googleapiclient
        from tabulate import tabulate


        credentials=Credentials.from_service_account_file('python-350618-3d9367733f29.json')

        # Set the keywords and time frame for the query
        kw_list = ["driving school", "learners license", "driving license",
                   "driving schools near me", "code 14 driving school"]
        timeframe = "now 1-d"  # Last one day only from now, it can take value 1 or 7 only

        # Create a pytrends object
        pytrends = TrendReq()

        # Set the geo and gprop parameters to get data for South African provinces
        pytrends.build_payload(kw_list, timeframe=timeframe, geo="ZA", gprop="")

        # Get the data
        df_regions = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=True)
        df_regions = df_regions.reset_index()


        # Export to google sheets

        file = 'python-350618-3d9367733f29.json'
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

        
        # post message to slack
        post_to_slack(f"""
                      :white_check_mark: Successful!
                       driving school: {df_regions['driving school'].sum()}
                       learners license: {df_regions['learners license'].sum()}
                       driving license: {df_regions['driving license'].sum()}
                       driving schools near me: {df_regions['driving schools near me'].sum()}
                       code 14 driving school: {df_regions['code 14 driving school'].sum()}
                      """
                      ,slack_credentials)
        
    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg, slack_credentials)


# In[ ]:




