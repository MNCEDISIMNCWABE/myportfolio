import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import certifi
import ssl

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the reports
base_url = 'https://pollencount.co.za/report/'

# Get Slack token and channel ID from environment variables
slack_bot_token = os.getenv('SLACK_BOT_TOKEN')
channel_id = os.getenv('SLACK_CHANNEL_ID_POLLEN')

# Initialize Slack client
slack_client = WebClient(token=slack_bot_token, ssl=ssl.create_default_context(cafile=certifi.where()))

def get_latest_friday():
    """
    Get the most recent Friday date.
    """
    today = datetime.today()
    # Find the most recent Friday
    last_friday = today - timedelta(days=(today.weekday() - 4) % 7)
    return last_friday

def fetch_report(url):
    """
    Fetch the HTML content of the report from the given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching the report: {e}")
        return None

def parse_pollen_level(soup):
    """
    Parse the pollen level and action for Cape Town from the HTML content.
    """
    try:
        # Find the 'Cape Town' section
        cape_town_section = soup.find('h5', text='Cape Town')
        if not cape_town_section:
            logging.error("Cape Town section not found.")
            return None, None
        
        # Find the parent row of the Cape Town section
        cape_town_row = cape_town_section.find_parent('div', class_='row')
        if not cape_town_row:
            logging.error("Cape Town row not found.")
            return None, None

        # Find the div containing the pollen level class (e.g., 'pollen-lightorange')
        pollen_div = cape_town_row.find('div', class_=lambda x: x and 'pollen-' in x)
        if not pollen_div:
            logging.error("Pollen level class not found for Cape Town.")
            return None, None

        pollen_class_name = pollen_div['class'][0]
        
        # Map the class name to the corresponding pollen level
        pollen_levels = {
            'pollen-verylow': 'Very Low',
            'pollen-low': 'Low',
            'pollen-yellow': 'Low',
            'pollen-lightorange': 'Moderate',
            'pollen-orange': 'High',
            'pollen-darkorange': 'High',
            'pollen-red': 'Very High',
            'pollen-green': 'Very Low'
        }

        # Map actions based on the pollen level
        actions = {
            'Very Low': "No action required. Pollen levels pose no risk to allergy sufferers.",
            'Low': "< 20% of pollen allergy sufferers will experience symptoms. Known seasonal allergy sufferers should commence preventative therapies e.g. nasal steroid sprays.",
            'Moderate': "> 50% of pollen allergy sufferers will experience symptoms. Need for increased use of acute treatments e.g. non-sedating antihistamines.",
            'High': "> 90% of pollen allergy sufferers will experience symptoms. Very allergic patients and asthmatics should limit outdoor activities and keep indoor areas free from wind exposure. Check section on pollen and day-to-day weather changes for planning activities.",
            'Very High': "These levels are potentially very dangerous for pollen allergy sufferers, especially asthmatics. Outdoor activities should be avoided."
        }

        # Get the level and the action based on the class name
        pollen_level = pollen_levels.get(pollen_class_name, 'Unknown')
        action = actions.get(pollen_level, "No specific action available for this level.")
        
        return pollen_level, action

    except Exception as e:
        logging.error(f"Error parsing pollen level: {e}")
        return None, None

def post_to_slack(city, date, pollen_level, action, report_url):
    """
    Post the pollen information to the Slack channel.
    """
    try:
        message = (
            f"*Pollen Report for {city}*\n"
            f"Date: {date}\n"
            f"Pollen Level: {pollen_level}\n"
            f"Action: {action}\n"
            f"Report URL: {report_url}"
        )
        response = slack_client.chat_postMessage(channel=channel_id, text=message)
        logging.info(f"Message sent to Slack: {response['ts']}")
        return True
    except SlackApiError as e:
        logging.error(f"Error sending message to Slack: {e.response['error']}")
        return False

def main():
    # Get the latest Friday
    latest_friday = get_latest_friday()
    latest_friday_formatted = latest_friday.strftime('%d-%B-%Y').lower()

    # Construct the URL for the latest report
    latest_report_url = f'{base_url}{latest_friday_formatted}/'
    print(f"Constructed URL for the latest report: {latest_report_url}")

    # Fetch the report content
    report_content = fetch_report(latest_report_url)
    if not report_content:
        return

    # Parse the HTML content
    soup = BeautifulSoup(report_content, 'html.parser')

    # Get the pollen level and action
    pollen_level, action = parse_pollen_level(soup)

    # Display the result if available and post to Slack
    if pollen_level and action:
        logging.info(f"Cape Town Overall Pollen Risk for {latest_friday_formatted}: {pollen_level}")
        logging.info(f"Action: {action}")

        # Post the information to Slack
        success = post_to_slack("Cape Town", latest_friday_formatted, pollen_level, action, latest_report_url)
        if success:
            logging.info("Pollen report posted to Slack successfully.")
        else:
            logging.error("Failed to post the pollen report to Slack.")
    else:
        logging.error("Pollen level and action could not be determined.")

if __name__ == "__main__":
    main()
