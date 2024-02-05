import pandas as pd
import requests
from datetime import datetime, timedelta
import pygsheets
import traceback

# Slack webhook URL
slack_webhook_url = 'https://hooks.slack.com/services/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'
# Service account file
service_account_file = 'bright-arc-328707-b5e2d782b48b.json'
# Google Sheet credentials
google_sheet_id = '1H7ANVCpN2bvkUpe8yIrsIu4rI6xC1z-61q83e40xNSw'
google_sheet_name = 'Bought'

# Function to authenticate and return a Google Sheet client
def authenticate_google_sheets(service_account_file):
    client = pygsheets.authorize(service_account_file=service_account_file)
    return client

# Function to fetch the spreadsheet data and return it as a pandas DataFrame
def get_spreadsheet_data(client, google_sheet_id, google_sheet_name):
    sheet = client.open_by_key(google_sheet_id)
    worksheet = sheet.worksheet_by_title(google_sheet_name)
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# Function to post a message to Slack
def post_to_slack(company_info):
    google_sheet_url = 'https://docs.google.com/spreadsheets/d/1H7ANVCpN2bvkUpe8yIrsIu4rI6xC1z-61q83e40xNSw/'
    message = f":bell: *Reminder*: Sell shares for *{company_info['Company']} _(in the next 7 days)_* :chart_with_upwards_trend:\n" \
              f"*Sell By Date:* {company_info['Sell By Date']} _(share price predicted to be highest on this date)_\n" \
              f"*Share Price At Buy:* {company_info['Share Price At Buy']}\n" \
              f"*Nr Of Shares Purchased:* {company_info['Nr Of Shares']}\n" \
              f"*Shares Value At Buy:* {company_info['Shares Value']}\n" \
              f"*Predicted Price:* {company_info['Predicted Price']}\n" \
              f"*Predicted Value:* {company_info['Predicted Value']}\n" \
              f"*Predicted Profit:* {company_info['Predicted Profit']}\n" \
              f"See more here: <{google_sheet_url}|Shares portfolio>"
    
    data = {"text": message}
    response = requests.post(slack_webhook_url, json=data)
    if response.status_code != 200:
        raise ValueError(f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}")


# Function to check dates and send reminders if necessary
def check_dates_and_remind(df):
    for index, row in df.iterrows():
        sell_by_date_str = row.get('Sell By Date', '')
        if sell_by_date_str:
            sell_by_date = datetime.strptime(sell_by_date_str, '%m/%d/%Y')  # Adjust the date format if necessary
            reminder_date = sell_by_date - timedelta(days=7)
            today = datetime.today()
            
            # Check if today is the reminder date
            if today.date() == reminder_date.date():
                post_to_slack(row)

# Main function that runs the whole process
def main():
    try:
        client = authenticate_google_sheets(service_account_file)
        df = get_spreadsheet_data(client, google_sheet_id, google_sheet_name)
        check_dates_and_remind(df)
    except Exception as e:
        # print the type of Exception, the Exception message, and the traceback
        error_message = f"An error occurred: {e}\n{traceback.format_exc()}"
        print(error_message)

if __name__ == "__main__":
    main()
