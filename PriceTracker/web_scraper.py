import requests
from bs4 import BeautifulSoup
from constants import *

def scrape_incredible_discounts():
    response = requests.get(f'{incredible_endpoint}{search_query}')
    soup = BeautifulSoup(response.text, 'lxml')
    apple_watches = soup.select('div.product-item-info')

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
                item_info = {
                    'Title': product_name,
                    'Link': product_url,
                    'Decrease': percentage_decrease,
                    'Special Price': special_price,
                    'Old Price': old_price
                }
                items.append(item_info)

    return items
