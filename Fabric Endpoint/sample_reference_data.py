# Sample reference data
countries = ["Ivory Coast", "South Africa", "Cameroon", "Swazi land", "Zambia"]
continents = ["Africa"] * 5
country_codes = ["CI", "ZA", "CM", "SZ", "ZM"]
continent_codes = ["AF"] * 5
device_models = {
    "Samsung": {
        "os": "Android",
        "models": ["Galaxy S22", "Galaxy A53", "Galaxy Note 20", "Galaxy Z Flip"]
    },
    "Apple": {
        "os": "iOS",
        "models": ["iPhone 13", "iPhone 14 Pro", "iPhone SE", "iPhone 12 Mini"]
    },
    "Huawei": {
        "os": "Android",
        "models": ["P50 Pro", "Mate 40", "Y9a", "Nova 9"]
    },
    "Xiaomi": {
        "os": "Android",
        "models": ["Redmi Note 11", "Mi 11", "Poco X4", "Redmi 10C"]
    }
}

network_types = ["WIFI", "MOBILE", "OFFLINE", "ETHERNET", "UNKNOWN"]
carriers = ["MTN"]
battery_states = ["UNPLUGGED", "CHARGING"]
connectivity_status = ["connected", "disconnected"]
install_store = ["Google Play","Apple App Store","Huawei AppGallery","Samsung Galaxy Store","APKMirror","Direct Download"]
install_source = ["organic","facebook_ads","google_ads","referral_program","sms_campaign","push_notification","influencer_campaign"]
cellular_generations = ["2G", "3G", "4G", "5G", "UNKNOWN"]
verticals = [
    "Telco - App Exclusive Offers",
    "Telco - Recharges", 
    "Telco - Balances",
    "MoMo", 
    "Loyalty & Rewards", 
    "Telco - PUK Retrieval", 
    "Digital Content - Read & Post Articles", 
    "Digital Content - Listen Music", 
    "Digital Content - Watch Videos", 
    "User Profile",
    "Home Page",
    "Registration",
    "Impressions"
    ]
traffic_names = [
    "summer_promo",
    "back_to_school",
    "black_friday_2025",
    "valentines_campaign",
    "referral_program",
    "new_user_acquisition",
    "email_push_1",
    "retargeting_facebook",
    "organic_growth",
    "loyalty_rewards"
]
traffic_mediums = [
    "email",
    "cpc",       # cost per click (paid ads)
    "organic",   # unpaid
    "social",
    "affiliate",
    "referral",
    "sms",
    "push_notification",
    "banner"
]
traffic_sources = [
    "google",
    "facebook",
    "twitter",
    "instagram",
    "newsletter",
    "tiktok",
    "in_app_message",
    "partner_site",
    "whatsapp",
    "direct"
]
geo_data = {
    "Ivory Coast": {
        "city": "Abidjan",
        "country_subdivision": "Abidjan Autonomous District",
        "latitude": 5.35444,
        "longitude": -4.00167,
        "domain": "http://mtn.ci",
        "isp_name": "MTN COTE D'IVOIRE S.A"
    },
    "South Africa": {
        "city": "Johannesburg",
        "country_subdivision": "Gauteng",
        "latitude": -26.2041,
        "longitude": 28.0473,
        "domain": "http://mtn.co.za",
        "isp_name": "MTN SOUTH AFRICA"
    },
    "Cameroon": {
        "city": "Douala",
        "country_subdivision": "Littoral",
        "latitude": 4.0511,
        "longitude": 9.7679,
        "domain": "http://mtn.cm",
        "isp_name": "MTN CAMEROON"
    },
    "Swazi land": {
        "city": "Mbabane",
        "country_subdivision": "Hhohho",
        "latitude": -26.3167,
        "longitude": 31.1333,
        "domain": "http://mtn.sz",
        "isp_name": "MTN ESWATINI"
    },
    "Zambia": {
        "city": "Lusaka",
        "country_subdivision": "Lusaka Province",
        "latitude": -15.3875,
        "longitude": 28.3228,
        "domain": "http://mtn.zm",
        "isp_name": "MTN ZAMBIA"
    }
}