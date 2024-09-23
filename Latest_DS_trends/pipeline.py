import requests
from bs4 import BeautifulSoup
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
import certifi
import ssl
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get Slack token and channel ID from environment variables
slack_bot_token = os.getenv('SLACK_BOT_TOKEN')
channel_id = os.getenv('SLACK_CHANNEL_ID')
slack_client = WebClient(token=slack_bot_token, ssl=ssl.create_default_context(cafile=certifi.where()))

# limit number of articles to send to slack to 5
limit_articles = int(os.getenv('LIMIT_ARTICLES'))

def post_to_slack(message, success=True):
    title = "Trending in Data Science:"
    icon = ":rocket:" if success else ":warning:"
    formatted_message = f"{icon} *{title}*\n{message}"
    
    try:
        response = slack_client.chat_postMessage(channel=channel_id, text=formatted_message, verify=False)
        logging.info(f"Message sent to Slack: {response['ts']}")
    except SlackApiError as e:
        logging.error(f"Error sending message to Slack: {e.response['error']}")


# Function to convert UTC date to SAST
def convert_to_sast(utc_datetime_str):
    try:
        utc_datetime = datetime.strptime(utc_datetime_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        utc_zone = pytz.utc
        sast_zone = pytz.timezone('Africa/Johannesburg')
        
        # Convert to SAST timezone
        utc_datetime = utc_zone.localize(utc_datetime)
        sast_datetime = utc_datetime.astimezone(sast_zone)
        
        # Format the SAST datetime
        return sast_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception as e:
        logging.error(f"Error converting date: {e}")
        return utc_datetime_str  # Return the original if conversion fails


# Function to fetch articles from either trending or latest
def get_articles(url, source, limit=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    
    try:
        # Send a request to the website with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Parse the page content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract all articles on the page
        articles = soup.find_all('h3', class_='graf graf--h3 graf-after--figure graf--title')
        
        # Extract the title, link, and the publication date
        articles_list = []
        for article in articles[:limit]:  # Limit to 'limit' number of articles
            title = article.text.strip()
            link = article.find_parent('a')['href']
            full_link = link  # The link might already be a full Medium link
            
            # Find the nearest <time> tag to extract the publication date
            time_tag = article.find_previous('time')
            if time_tag:
                date_utc = time_tag['datetime']
                date_sast = convert_to_sast(date_utc)
            else:
                date_sast = 'No Date Available'
            
            articles_list.append({"title": title, "link": full_link, "date": date_sast, "source": source})
        
        return articles_list
    except requests.RequestException as e:
        logging.error(f"Error fetching {source} articles: {e}")
        return []


# Function to format articles for Slack message
def format_articles_for_slack(articles, source):
    message = f"*{source} Articles:*\n"
    for idx, article in enumerate(articles, 1):
        message += f"{idx}. *{article['title']}*\n   <{article['link']}|Read Article>\n   Date Published: `{article['date']}`\n"
    return message


# Fetch only 5 trending articles
trending_articles = get_articles("https://towardsdatascience.com/trending", "Trending", limit=limit_articles)

# Fetch only 5 latest articles
latest_articles = get_articles("https://towardsdatascience.com/latest", "Latest", limit=limit_articles)

# Combine both into a single list
all_articles = trending_articles + latest_articles

# Convert the combined list of dictionaries into a pandas DataFrame
df_all_articles = pd.DataFrame(all_articles)

# Format the Slack message for Trending articles
if trending_articles:
    trending_message = format_articles_for_slack(trending_articles, "Trending")
    post_to_slack(trending_message)
else:
    logging.info("No Trending articles found to post.")

# Format the Slack message for Latest articles
if latest_articles:
    latest_message = format_articles_for_slack(latest_articles, "Latest")
    post_to_slack(latest_message)
else:
    logging.info("No Latest articles found to post.")