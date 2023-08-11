from post_to_slack import post_to_slack
from google_trends import get_google_trends_data
from constants import *

if __name__ == '__main__':
    try:
        df_regions = get_google_trends_data(google_service_account_file)

        post_to_slack(f"""
            :white_check_mark: Successful!
            driving school: {df_regions['driving school'].sum()}
            learners license: {df_regions['learners license'].sum()}
            driving license: {df_regions['driving license'].sum()}
            driving schools near me: {df_regions['driving schools near me'].sum()}
            code 14 driving school: {df_regions['code 14 driving school'].sum()}
        """, slack_credentials, title)
    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg, slack_credentials, title)
