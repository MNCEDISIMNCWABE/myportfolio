from google.oauth2.service_account import Credentials
slack_credentials = 'https://hooks.slack.com/services/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'
google_service_account_file = Credentials.from_service_account_file('ornate-genre-425416-q8-39e4e509df0e.json')
file = 'bright-arc-328707-b5e2d782b48b.json'
google_sheet_id = '1wu6wT8GwPitzhY8Yov3FspFBTd2dC6znKKNcbwP70UQ'
google_sheet_name = 'share_price_forecast'
bq_project_id = 'bright-arc-328707'
bq_dataset_id = 'test'
bq_table_id = 'stock_price_predictions'
start_date = '2014-01-02'
ticker_list = ['CLS.JO', 'GLN.JO', 'WHL.JO', 'APN.JO', 'PIK.JO', 'PPE.JO',
               'HIL.JO', 'SOL.JO', 'EXX.JO', 'MCG.JO', 'AIL.JO', 'TGA.JO', 'SSW.JO', 'INL.JO',
               'SHP.JO','FSR.JO','OMU.JO','SLM.JO','RMH.JO','GRT.JO','DSY.JO','GFI.JO']
title = (f":rotating_light: Stock Price Predictions Run:")
