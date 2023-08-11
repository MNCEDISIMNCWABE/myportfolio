import requests

def post_to_slack(message, slack_credentials, title):
    data = {
        "icon_emoji": ":white_check_mark:",
        "attachments": [{
            "color": "#9733EE",
            "fields": [{"title": title, "value": message, "short": "false"}]
        }]
    }
    requests.post(slack_credentials, json=data, headers={'Content-Type': 'application/json'}, verify=True)
