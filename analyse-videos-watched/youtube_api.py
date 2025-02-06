import os
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

def authenticate_youtube():
    """Authenticate with YouTube Data API."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('youtube', 'v3', credentials=creds)

def get_liked_videos(youtube, max_results=50):
    """Fetch liked videos."""
    request = youtube.videos().list(
        part="snippet,contentDetails",
        myRating="like",
        maxResults=max_results
    )
    response = request.execute()
    return response.get('items', [])

def get_watched_videos(youtube, max_results=50):
    """Fetch watched videos."""
    # Note: The YouTube Data API does not provide direct access to watch history.
    # You can only access public data like liked videos, playlists, etc.
    # To access watch history, you need to use the YouTube History API (not part of the Data API).
    # This function is a placeholder and will not work as-is.
    raise NotImplementedError("YouTube Data API does not support fetching watched videos directly.")