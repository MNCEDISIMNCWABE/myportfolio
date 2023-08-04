# Send message to slack    
# incoming webhook
slack_credentials = 'https://hooks.slack.com/services/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'
title = (f":rotating_light: Price Tracker Run:")

import requests
 
def post_to_slack(message,slack_credentials):
    data = { "icon_emoji": ":white_check_mark:",
             "attachments": [{"color": "#9733EE","fields": [{"title": title,"value": message,"short": "false"}]}]}
    url = slack_credentials
    requests.post(url, json=data, headers={'Content-Type': 'application/json'}, verify=True)


if __name__ == '__main__':
    try:
        import requests
        from bs4 import BeautifulSoup

        # DECLARE CONSTANTS Apple = 5456, Samsung = 5457
        INCREDIBLE_ENDPOINT = 'https://www.incredible.co.za/'
        SEARCH_QUERY = 'products/appliances-new-9/laundry/front-loader-washing-machines'

        response = requests.get(f'{INCREDIBLE_ENDPOINT}{SEARCH_QUERY}')
        soup = BeautifulSoup(response.text, 'lxml')
        apple_watches = soup.select('div.product-item-info')

        # Empty list to store the data
        items = []

        for watch in apple_watches:
            url = watch.a['href']
            product_name = watch.select_one('a.product-item-link').text
            product_url = watch.select_one('a.product-item-link')['href']
            old_price_tag = watch.select_one('span.old-price span.price-wrapper  span.price')
            if old_price_tag is not None:
                old_price = float(watch.select_one('span.old-price span.price-wrapper  span.price').text[1:].replace(",", ""))
                special_price = float(watch.select_one('span.special-price span.price').text[1:].replace(",", ""))
                percentage_decrease = (old_price - special_price) / old_price * 100

                if percentage_decrease > 5:
                    # Add the relevant information for the item to the list
                    item_info = {
                        'Title': product_name,
                        'Link': product_url,
                        'Decrease': percentage_decrease,
                        'Special Price': special_price,
                        'Old Price': old_price
                    }
                    items.append(item_info)

                    # Post message to slack for each product
                    post_to_slack(f"""
                                  :white_check_mark: Successful!
                                   Product Name:{product_name}
                                   URL:{product_url}
                                   {f"Old price: R{old_price:,.2f}"}
                                   {f"Special price:R{special_price:,.2f}"}
                                   Decrease:{f"{percentage_decrease:.2f}%"}
                                  """
                                  , slack_credentials)
        # Check if there are no items with discounts
        if not items:
            post_to_slack("No discounts found.", slack_credentials)

        # Optionally, if you want to post a single message with all the products:
       #slack_message = "\n".join([f":white_check_mark: Successful!\nProduct Name: {item['Title']}\nURL: {item['Link']}\nOld price: R {item['Old Price']:.2f}\nSpecial price: R {item['Special Price']:.2f}\nDecrease: {item['Decrease']:.2f}%" for item in items])
       #post_to_slack(slack_message, slack_credentials)

    except:
        msg = 'Error in the script!'
        post_to_slack(msg, slack_credentials)