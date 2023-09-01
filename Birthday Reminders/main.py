import requests
from datetime import datetime
from constants import slack_credentials , people_birthdays 
from get_birthday_message import get_birthday_message
from get_birthdays import get_birthdays
from post_to_slack import post_to_slack

if __name__ == '__main__':
    try:
        birthdays = get_birthdays()
        
        today = datetime.today().strftime('%d-%m')  # Get today's date in 'dd-mm' format
        
        # Check if it's someone's birthday and send a personalized message
        for name in birthdays:
            message = get_birthday_message(name, today)
            if message:
                # Update the title to match the name of the person whose birthday it is
                title = f":birthday: Hey Ya'll, it's {name}'s birthday!!!"
                post_to_slack(title, message, slack_credentials)
        
    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg, slack_credentials)
