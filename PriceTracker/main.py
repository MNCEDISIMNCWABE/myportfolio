from slack_utils import post_to_slack
from web_scraper import scrape_incredible_discounts

if __name__ == '__main__':
    try:
        items = scrape_incredible_discounts()

        if items:
            for item in items:
                slack_message = (
                    f":white_check_mark: Successful!\n"
                    f"Product Name: {item['Title']}\n"
                    f"URL: {item['Link']}\n"
                    f"Old price: R {item['Old Price']:.2f}\n"
                    f"Special price: R {item['Special Price']:.2f}\n"
                    f"Decrease: {item['Decrease']:.2f}%"
                )
                post_to_slack(slack_message, slack_credentials, title)
        else:
            post_to_slack("No discounts found.", slack_credentials, title)
    except Exception as e:
        msg = f'Error in the script: {e}'
        post_to_slack(msg, slack_credentials, title)
