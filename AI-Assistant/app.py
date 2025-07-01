from flask import Flask, request, jsonify
from slack_sdk import WebClient
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai

app = Flask(__name__)

# Initialize Slack client with OAuth Access Token
slack_token = 'xoxp-3794251012583-3811267290740-7287821409286-c61ad37d147f25e21ea833ea6d787be7'
slack_client = WebClient(token=slack_token)

# Initialize Google Drive client
credentials = service_account.Credentials.from_service_account_file('/Users/mncedisimncwabe/Downloads/ornate-genre-425416-q8-0747992745db.json')
drive_service = build('drive', 'v3', credentials=credentials)

# Initialize OpenAI client
openai.api_key = 'sk-proj-Xwjm22bDTkANT3UhY9ziT3BlbkFJvSOEuEUIRePXFGPz6JLO'

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    response = get_answer(question)
    return jsonify(response)

def get_answer(question):
    # Search in Slack
    slack_response = search_slack(question)
    
    # Search in Google Drive
    drive_response = search_drive(question)

    # Generate a human-like response
    response = generate_human_response(question, slack_response, drive_response)
    
    return response

def search_slack(query):
    result = slack_client.search_messages(query=query)
    messages = []
    for message in result['messages']['matches']:
        text = message['text']
        user = message['user']
        permalink = message['permalink']
        messages.append({
            "text": text,
            "user": user,
            "permalink": permalink
        })
    return messages

def search_drive(query):
    results = drive_service.files().list(q=f"name contains '{query}'", fields="files(id, name, webViewLink)").execute()
    files = [{"name": file['name'], "link": file['webViewLink']} for file in results.get('files', [])]
    return files

def generate_human_response(question, slack_response, drive_response):
    context = ""

    if slack_response:
        for message in slack_response:
            context += f"Message: {message['text']}\nPosted by: {message['user']}\nLink: {message['permalink']}\n\n"
    
    if drive_response:
        for file in drive_response:
            context += f"File: {file['name']}\nLink: {file['link']}\n\n"
    
    if not context:
        context = "No relevant information found."

    # Use OpenAI's GPT-3 to generate a human-like response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        max_tokens=150
    )
    
    answer = response.choices[0].text.strip()

    return {"response": answer}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)