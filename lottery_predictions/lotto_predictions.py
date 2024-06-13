import pandas as pd
import numpy as np
import requests
import datetime as dt
import os
import random
from tabulate import tabulate
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

slack_credentials = 'https://hooks.slack.com/services/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'
title = (f":rotating_light: Lottery Predictions:")


def post_to_slack(message,slack_credentials):
    data = { "icon_emoji": ":white_check_mark:",
             "attachments": [{"color": "#9733EE","fields": [{"title": title,"value": message,"short": "false"}]}]}
    url = slack_credentials
    requests.post(url, json=data, headers={'Content-Type': 'application/json'}, verify=True)
    
    
    
#get data function
def combine_lotto_data(folder_path, years):
    dfs = []

    # Loop through each year, read the corresponding file, and append to the list
    for year in years:
        file_path = os.path.join(folder_path, f"daily_lotto_{year}.txt")
        df = pd.read_csv(file_path, delimiter='\t', header=0)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Rename the columns
    combined_df.columns = ['Date', 'Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']

    return combined_df

#remove nulls function
def drop_nulls(df):

    df = df.drop('Number6', axis=1)

    return df

#convert date to datetime
def convert_to_datetime(df, date_column):

    df[date_column] = pd.to_datetime(df[date_column])
    return df

#extract datetime components
def extract_date_time_components(df):

    # Extract Hour, Day of the month, and Day of the week
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week

    return df

#train and predict
def predict_next_7_days_lottery_numbers(combined_df, features, target_numbers):
    # Prepare data
    X = combined_df[features]
    y = combined_df[target_numbers]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction for the Next 7 Days
    today = datetime.now()
    next_7_days_features = [{
        'DayOfMonth': (today + timedelta(days=i)).day,
        'Week': (today + timedelta(days=i)).isocalendar()[1]
    } for i in range(7)]

    # Initialize DataFrame for predicted numbers
    predicted_numbers = pd.DataFrame(columns=['Date'] + target_numbers)

    # Generate predictions for each day
    for i in range(7):
        day_features_df = pd.DataFrame([next_7_days_features[i]])
        attempts = 0

        while attempts < 200:  # Adjust the maximum attempts as needed
            day_prediction = model.predict(day_features_df)
            day_prediction = np.clip(day_prediction.round(), 1, 36)  # Round and clamp

            # Check for uniqueness
            if len(set(day_prediction[0])) == len(day_prediction[0]):
                break  # Break out of the loop if unique
            attempts += 1

        # If unique numbers are not found, add a row with the last prediction
        if attempts == 200:
            day_prediction = np.random.randint(1, 37, size=len(target_numbers))

        # Convert the prediction to a list and flatten it
        day_prediction_list = day_prediction.flatten().tolist()

        # Add the date and predicted numbers to the DataFrame
        predicted_row = pd.DataFrame([[today + timedelta(days=i)] + day_prediction_list], columns=['Date'] + target_numbers)
        predicted_numbers = pd.concat([predicted_numbers, predicted_row], ignore_index=True)

    return predicted_numbers

    
if __name__ == '__main__':
    try:    

        ###---- call functions----
        #1--get data
        folder_path = "lottery_predictions"
        years = ['2019', '2020', '2021', '2022', '2023']
        combined_df = combine_lotto_data(folder_path, years)
        #2--drop nulls
        combined_df = drop_nulls(combined_df)
        #3--convert to datetime
        combined_df = convert_to_datetime(combined_df,'Date')
        #4--extract datetime components
        combined_df = extract_date_time_components(combined_df)
        #5--predictions
        features = ['DayOfMonth', 'Week']
        target_numbers = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5']
        predicted_results = predict_next_7_days_lottery_numbers(combined_df, features, target_numbers)
        predicted_results['Date'] = predicted_results['Date'].dt.strftime('%Y-%m-%d')

        predicted_results.reset_index(drop=True, inplace=True)

        # Convert 'predicted_results' to a table string format for Slack
        formatted_results = tabulate(predicted_results, showindex=False, headers=['Date', 'Number1', 'Number2', 'Number3', 'Number4', 'Number5'], tablefmt='pipe')


        # post message to slack
        post_to_slack(f"""
                      :white_check_mark: Successful!
                       Predictions:\n```{predicted_results}```
                      """
                      ,slack_credentials)

    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg, slack_credentials)
