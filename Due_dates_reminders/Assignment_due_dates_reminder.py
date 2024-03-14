import requests
from datetime import datetime, timedelta

slack_credentials = 'https://hooks.slack.com/services/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'

def post_to_slack(title: str, message: str, slack_credentials: str) -> None:
    # Format the message to include the <!channel> tag which Slack recognizes for notifications.
    formatted_message = f"<!channel>\n"
    data = {
        "text": formatted_message,  # This field is the main message text
        "attachments": [{
            "color": "#9733EE",
            "fields": [{
                "title": title,
                "value": message,
                "short": "false"
            }]
        }]
    }
    requests.post(slack_credentials, json=data, headers={'Content-Type': 'application/json'}, verify=True)

def send_reminder(dates_titles: dict, slack_credentials: str) -> None:
    now = datetime.now()
    for date_str, title in dates_titles.items():
        due_date = datetime.strptime(date_str, '%B %d')
        due_date = due_date.replace(year=now.year)

        intervals = {
            "5 days": timedelta(days=5),
            "3 days": timedelta(days=3),
            "1 day": timedelta(days=1)
        }

        # Sort the intervals to check the smallest (most imminent) first
        for reminder_time, tdelta in sorted(intervals.items(), key=lambda x: x[1]):
            # Check if today is exactly the reminder interval before the due date
            if due_date - tdelta == now.replace(hour=0, minute=0, second=0, microsecond=0):
                message = f"Reminder: {title} in {reminder_time}."
                post_to_slack(":rotating_light: Upcoming Due Date :rotating_light:", message, slack_credentials)
                break

if __name__ == '__main__':
    dates_titles = {
        "March 19": "Dispute resolution 1 closes",
        "March 27": "Opinion 2 Assignment releases",
        "March 28": "Knowledge assessment closes",
        "April 5" : "Knowledge Assignment 3 releases",
        "April 22": "Opinion 2 Submission",
        "April 26": "Dispute Resolution 2 Assignment releases",
        "May 2"   : "Knowledge assessment 3 submission",
        "May 3"   : "Knowledge Assignment 4 releases",
        "May 20"  : "Dispute Resolution 2 submission",
        "May 24"  : "Opinion 3 Assignment releases",
        "May 31"  : "Knowledge assessment 4 submission",
        "June 7"  : "Knowledge Assignment 5 releases",
        "June 18" : "Opinion 3 submission",
        "June 22" : "Dispute Resolution 3 Assignment releases",
        "June 28" : "Knowledge assignment 5 submission",
        "July 2"  : "Registration documents for EISA to SAIT submission",
        "July 5"  : "Knowledge Assignment 6 releases",
        "July 15" : "Dispute Resolution submission",
        "July 16" : "Hot Topix 1 releases",
        "July 24" : "Hot Topix 2 releases",
        "July 30" : "Knowledge Assignment 6 submission",
        "August 3": "Mock exam day 1",
        "August 4": "Mock exam day 2"
    }

    send_reminder(dates_titles, slack_credentials)

