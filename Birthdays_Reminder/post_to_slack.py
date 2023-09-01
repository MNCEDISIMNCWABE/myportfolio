import requests

def post_to_slack(title: str, message: str, slack_credentials: str) -> None:
    data = {
        "icon_emoji": ":white_check_mark:",
        "attachments": [{"color": "#9733EE", "fields": [{"title": title, "value": message, "short": "false"}]}]
    }
    url = slack_credentials
    requests.post(url, json=data, headers={'Content-Type': 'application/json'}, verify=True)
