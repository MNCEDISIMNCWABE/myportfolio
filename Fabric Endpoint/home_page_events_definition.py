import datetime 
import random
from sample_reference_data import verticals

start_date = datetime.datetime(2025, 1, 1)
event_date = start_date + datetime.timedelta(days=random.randint(0, 364))
session_start = event_date + datetime.timedelta(seconds=random.randint(0, 86400))
start_timestamp = int(session_start.timestamp())
user_id = f"+2771{random.randint(10000000, 99999999)}"

home_page_events = [
    {
        "event_name": "home_open",
        "entry_point": "homepage",
         "vertical": "Home Page",
        "event_params": {
        }
    },
    {
        "event_name": "click_quicklink",
        "entry_point": "homepage",
        "vertical": "Home Page",
        "event_params": {
            "category": random.choice(["balances", "rewards", "made-4u"])
        }
    },
    {
        "event_name": "home_balance_view",
        "entry_point": "homepage",
        "vertical": "Telco - Recharges",
        "event_params": {
            "balance_type": {
                "airtime": 200,
                "data": "20 GB",
                "voice": "100 mins",
                "sms": "50 sms",
                "momo": None
            }
        }
    },
    {
        "event_name": "click_vertical",
		"entry_point": "homepage",
		"vertical":  random.choice(verticals),
        "event_params": {
        }
    },
    {
        "event_name": "quick_recharge",
		"entry_point": "homepage",
		"vertical": "Telco - Recharges",
        "event_params": {
        }
    },
    {
        "event_name": "channelcard_impressions_end",
		"entry_point": "homepage",
		"vertical": "Digital Content - Read & Post Articles",
        "event_params": {
            "content": {
                "channel_name": "ABC",
                "channel_publication_name": "abcs",
                "channel_id": 1,
                "card_id": 2,
                "channel_publication_date": "2024-12-12 12:00:00"
            },
            "impression_id": f"{user_id}_{start_timestamp}",
            "card_state": "collapsed"  # This can also be expanded
        }
    },
    {
        "event_name": "channelcard_impressions_start",
		"entry_point": "homepage",
		"vertical": "Digital Content - Read & Post Articles",
        "event_params": {
            "content": {
                "channel_name": "ABC",
                "channel_publication_name": "abcs",
                "channel_id": 1,
                "card_id": 2,
                "channel_publication_date": "2024-12-12 12:00:00"
            },
            "impression_id": f"{user_id}_{start_timestamp}",  # where start_timestamp is the equivalent of event_timestamp
            "card_state": "collapsed"  # This can also be expanded
        }
    },
    {
        "event_name": "delete_quicklink",
		"entry_point": "homepage",
		"vertical":  random.choice(verticals),
        "event_params": {
            "quicklink": random.choice(["recharge", "bookmark", "listen", "rewards", "read", "watch"]),
            "position": 3,
            "deleted_by": random.choice(["user", "admin"])  # This can also be admin 
        }
    },
    {
        "event_name": "click_chat",
		"entry_point": "homepage",
		"vertical": "Digital Content - Read & Post Articles",
        "event_params": {
            "recent_chats": [{
                "user_name": "ay",
                "user_id": "+277888181"
            }]
        }
    },
    {
        "event_name": "click_promo",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "promo_id": 17272,
            "web_link": "www.disney.com"
        }
    },
    {
        "event_name": "view_promo",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "promo_id": 17272,
            "web_link": "www.disney.com",
        }
    },
    {
        "event_name": "click_competition",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "competition_id": 17272
        }
    },
    {
        "event_name": "click_search",
		"entry_point": "homepage",
		"vertical": "Digital Content - Read & Post Articles",
        "event_params": {
            "search_id": 17272
        }
    },
    {
        "event_name": "click_search_result",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "search_term": "data",
            "search_length": 4,
            "search_response": {
                "games": [], 
                "music": []
            }
        }
    },
    {
        "event_name": "click_user_profile",
		"entry_point": "homepage",
		"vertical": "User Profile",
        "event_params": {
            "user_profile_id": 1234
        }
    },
    {
        "event_name": "notifications_banner_event",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "notifications_banner_id": 1234
        }
    },
    {
        "event_name": "click_notification_dismiss",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "notification_id": 1234,
            "unread_messages_count": 5  # This can be any number
        }
    },
    {
        "event_name": "notification_check_messages",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "notification_id": 1234,
            "unread_messages_count": 5  # This can be any number
        }
    },
    {
        "event_name": "scroll",
		"entry_point": "homepage",
		"vertical": "Home Page",
        "event_params": {
            "engagement_time_msec": 1313121,
            "percent_scrolled": 10  # This can range from 50 to 100
        }
    },
    {
        "event_name": "click_notification_dismiss",
		"entry_point": "homepage",
		"vertical": "Telco - App Exclusive Offers",
        "event_params": {
            "unread_messages_count": 5
        }
    },
    {
        "event_name": "registration_process_started",
		"entry_point": "homepage",
		"vertical": "Registration",
        "event_params": {
            "method": "he_unavailable",  # The registration method like social, he, he_unavailable (OTP)
            "type": "he_unavailable"  # The registration method like social, he, he_unavailable (OTP)
        }
    },
    {
        "event_name": "login",
		"entry_point": "homepage",
		"vertical": "Registration",
        "event_params": {
            "method": "he_unavailable"
        }
    },
    {
        "event_name": "view_story",
		"entry_point": "homepage",
		"vertical": "Digital Content - Watch Videos",
        "event_params": {
            "story_id": 1234,
            "story_type": "video",  # This can also be image, text, etc.
            "duration": 5000  # Duration in milliseconds
        }
    }
]