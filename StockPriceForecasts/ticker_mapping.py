import pandas as pd

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
        'SHP.JO':'Shoprite,
        'CFR.JO': 'Richemont',
        'FSR.JO':'FirstRand',
        'OMU.JO','Old-Mutual',
        'SLM.JO': 'Sanlam',
        'RMH.JO':'RMB',
        'GRT.JO':'Growth-Point',
        'DSY.JO':'Discovery',
        'GFI.JO':'Gold-Fields',
        'SNH.JO':'Steinhoff'
    }

    for_loop_forecast['ticker_name'] = for_loop_forecast['ticker'].map(ticker_name_mapping)

    return for_loop_forecast
