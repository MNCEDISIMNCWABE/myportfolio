slack_credentials = 'https://hooks.slack.com/services/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'
title = (f":rotating_light: Driving School Search Terms - Google Trends Past 24 hours:")
# Set the keywords and time frame for the query
kw_list = ["driving school", "learners license", "driving license", "driving schools near me", "code 14 driving school"]
timeframe = "now 1-d"  # Last one day from now, it can take value 1 or 7 only
# Export to google sheets
file = 'python-350618-3d9367733f29.json'
sheet_id = '1TDg0Gb_M5LFd-zOi70nmw2dyCpACqMi5qLDJTl4xxFk' 
sheet_name = 'trends'
