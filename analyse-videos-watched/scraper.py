import subprocess
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

api_key = "23b78375c0dcb16ea17bd05bafe20f74e8bc966640008f86e3eeedf3455aac97"

def trigger_scraping_channels(api_key, channel_urls, num_of_posts, start_date, end_date, order_by, country):
    dataset_id = "gd_lk56epmy2i5g7lzu0k"
    endpoint = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={dataset_id}&include_errors=true&type=discover_new&discover_by=url"

    payload = [
        {
            "url": url,
            "num_of_posts": num_of_posts,
            "start_date": start_date,
            "end_date": end_date,
            "order_by": order_by,
            "country": country
        }
        for url in channel_urls
    ]

    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
        endpoint
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return None
    else:
        print(f"Error: {result.stderr}")
        return None

def get_progress(api_key, snapshot_id):
    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout.strip())
    else:
        print(f"Error: {result.stderr}")
        return None

def get_output(api_key, snapshot_id, format="json"):
    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format={format}"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        json_lines = result.stdout.strip().split("\n")
        json_objects = [json.loads(line) for line in json_lines]
        return json_objects
    else:
        print(f"Error: {result.stderr}")
        return None