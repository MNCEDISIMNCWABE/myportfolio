from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
import logging
import warnings
import re
from typing import List, Optional
import io
import hashlib
import pickle
import os
from datetime import datetime, timedelta
from flask import send_file 
from flask_session import Session

app = Flask(__name__)
app.secret_key = '1998f4d5ac07df2aa1a2a5f22d2b9f87'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app) 

app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['SESSION_TYPE'] = 'filesystem'

# Set up logging and ignore warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User authentication functions
def make_hashed_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(stored_password, input_password):
    return stored_password == make_hashed_password(input_password)

def save_users(users_dict):
    with open('users.pkl', 'wb') as f:
        pickle.dump(users_dict, f)

def load_users():
    if os.path.exists('users.pkl'):
        with open('users.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        users = {
            'admin': {
                'password': make_hashed_password('SPI123@_'),
                'email': 'admin@example.com',
                'created_at': datetime.now(),
                'role': 'admin'
            }
        }
        save_users(users)
        return users

# API functions
def search_employees_one_row_per_employee_dedup(query, country_filter=None, location_filter=None, max_to_fetch=1, exclude_patterns=None):
     # Set default exclude patterns if none provided
    if exclude_patterns is None:
        exclude_patterns = ["PA to", "Assistant to", "Personal Assistant"]


    must_clauses = []

    # a) The nested query for experience titles
    must_clauses.append({
        "nested": {
            "path": "member_experience_collection",
            "query": {
                "query_string": {
                    "query": query,
                    "default_field": "member_experience_collection.title",
                    "default_operator": "and"
                }
            }
        }
    })

    # b) If user wants to filter by a specific country (exact match)
    if country_filter:
        must_clauses.append({
            "term": {
                "country": country_filter
            }
        })

    # c) If user wants to filter by a specific location (phrase match)
    if location_filter:
        must_clauses.append({
            "match_phrase": {
                "location": location_filter
            }
        })

    # Combine into a bool query
    payload = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }

    # 2) Send the search request
    search_url = "https://api.coresignal.com/cdapi/v1/professional_network/employee/search/es_dsl"
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJFZERTQSIsImtpZCI6IjEwYTYwZWRhLWNhNzEtMTIxZS1jY2JhLTBmNjRjMzg4Yjg0ZCJ9.eyJhdWQiOiJheW9iYS5tZSIsImV4cCI6MTc3MzEwNjAyMSwiaWF0IjoxNzQxNTQ5MDY5LCJpc3MiOiJodHRwczovL29wcy5jb3Jlc2lnbmFsLmNvbTo4MzAwL3YxL2lkZW50aXR5L29pZGMiLCJuYW1lc3BhY2UiOiJyb290IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYXlvYmEubWUiLCJzdWIiOiI5Nzg4ZDg5Ni0yNzBjLTU4NjgtMTY0Mi05MWFiZDk0MGEwODYiLCJ1c2VyaW5mbyI6eyJzY29wZXMiOiJjZGFwaSJ9fQ.BeR_ci_7346iPkfP64QZCwxILa1v1_HGIE1SdhOl9qHtM_HcwiiWIf26DNhcDPl7Bs16JAEfjBntMoyJymtYDA'
    }
    try:
        resp = requests.post(search_url, headers=headers, json=payload)
        resp.raise_for_status()
        employee_ids = resp.json()

        if not isinstance(employee_ids, list):
            logger.error("Unexpected structure in search response")
            return pd.DataFrame()

        # 3) Collect data for each employee ID
        rows = []
        for emp_id in employee_ids[:max_to_fetch]:
            collect_url = f"https://api.coresignal.com/cdapi/v1/professional_network/employee/collect/{emp_id}"
            r = requests.get(collect_url, headers=headers)
            r.raise_for_status()

            employee = r.json()

            # Basic fields
            id_val = employee.get('id')
            name_val = employee.get('name')
            headline_val = employee.get('title')
            location_val = employee.get('location')
            country_val = employee.get('country')
            url_val = employee.get('url')
            canonical_url = employee.get('canonical_url')
            industry_val = employee.get('industry')
            experience_count_val = employee.get('experience_count')
            summary_val = employee.get('summary')

            # ----- EXPERIENCE (deduplicate) -----
            raw_exps = employee.get('member_experience_collection', [])
            unique_exps = []
            seen_exps = set()
            for exp in raw_exps:
                key = (
                    exp.get('title', 'N/A'),
                    exp.get('company_name', 'N/A'),
                    exp.get('date_from', 'N/A'),
                    exp.get('date_to', 'N/A')
                )
                if key not in seen_exps:
                    seen_exps.add(key)
                    unique_exps.append(exp)

            experiences_str = "\n".join(
                f"Role: {exp.get('title','N/A')} | Company: {exp.get('company_name','N/A')} "
                f"| From: {exp.get('date_from','N/A')} | To: {exp.get('date_to','N/A')} "
                f"| Duration: {exp.get('duration','N/A')}"
                for exp in unique_exps
            )

            # ----- EDUCATION (deduplicate) -----
            raw_edu = employee.get('member_education_collection', [])
            unique_edu = []
            seen_edu = set()
            for edu in raw_edu:
                key = (
                    edu.get('title', 'N/A'),
                    edu.get('subtitle', 'N/A'),
                    edu.get('date_from', 'N/A'),
                    edu.get('date_to', 'N/A')
                )
                if key not in seen_edu:
                    seen_edu.add(key)
                    unique_edu.append(edu)

            educations_str = "\n".join(
                f"Institution: {edu.get('title','N/A')} | Degree: {edu.get('subtitle','N/A')} "
                f"| From: {edu.get('date_from','N/A')} | To: {edu.get('date_to','N/A')}"
                for edu in unique_edu
            )

            # ----- SKILLS (deduplicate) -----
            raw_skills = employee.get('member_skills_collection', [])
            seen_skills = set()
            for skill_entry in raw_skills:
                skill_name = skill_entry.get('member_skill_list', {}).get('skill', 'N/A')
                if skill_name not in seen_skills:
                    seen_skills.add(skill_name)

            skills_str = ", ".join(seen_skills) if seen_skills else ""

            # Build final row
            row = {
                "ID": id_val,
                "Name": name_val,
                "Headline/Title": headline_val,
                "Location": location_val,
                "Country": country_val,
                "URL": url_val,
                "Canonical_URL": canonical_url,
                "Industry": industry_val,
                "Experience Count": experience_count_val,
                "Summary": summary_val,
                "Experiences": experiences_str,
                "Educations": educations_str,
                "Skills": skills_str
            }
            rows.append(row)

        # After collecting all rows, before returning the dataframe, add this filtering:
        filtered_rows = []
        for row in rows:
            # Check if the headline/title contains any of the exclude patterns
            should_exclude = False
            headline = row.get("Headline/Title", "")
            if headline:
                for pattern in exclude_patterns:
                    if pattern.lower() in headline.lower():
                        should_exclude = True
                        break
                        
            # Also check experiences for titles containing exclude patterns
            experiences = row.get("Experiences", "")
            if experiences and not should_exclude:
                for pattern in exclude_patterns:
                    # Only match pattern in the "Role:" part of experiences
                    role_pattern = f"Role: {pattern}"
                    if role_pattern.lower() in experiences.lower():
                        should_exclude = True
                        break
            
            # If not excluded, add to filtered results
            if not should_exclude:
                filtered_rows.append(row)
        
        df = pd.DataFrame(filtered_rows)
        return df

    except Exception as e:
        logger.error(f"Error in search_employees: {str(e)}")
        return pd.DataFrame()

# Ranking functions
def build_user_text(row, text_columns: List[str]) -> str:
    parts = []
    for col in text_columns:
        val = row.get(col)
        if pd.notnull(val):
            if isinstance(val, list):
                parts.append(' '.join(map(str, val)))
            else:
                parts.append(str(val))
    return " ".join(parts).strip()

def preprocess_text(text: str) -> str:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(text.strip().split())
    return text

def rank_candidates_semantic(df_employees: pd.DataFrame, job_description: str, text_columns: Optional[List[str]] = None, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32) -> pd.DataFrame:
    try:
        logger.info("Starting candidate ranking process...")

        df = df_employees.copy()

        if df.empty:
            logger.warning("Empty dataframe provided for ranking")
            return pd.DataFrame()

        if text_columns is None:
            text_columns = ['Summary', 'Experiences', 'Educations',
                           'Headline/Title', 'Industry', 'Skills']

        df['combined_text'] = df.apply(
            lambda x: build_user_text(x, text_columns),
            axis=1
        )

        df['combined_text'] = df['combined_text'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['combined_text']).reset_index(drop=True)

        if df.empty:
            logger.warning("No valid candidate texts found after preprocessing")
            return pd.DataFrame()

        model = SentenceTransformer(model_name)

        clean_jd = preprocess_text(job_description)
        job_embedding = model.encode(clean_jd, convert_to_tensor=True)

        user_texts = df['combined_text'].apply(preprocess_text).tolist()
        user_embeddings = model.encode(
            user_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=True
        )

        similarities = util.cos_sim(job_embedding, user_embeddings)
        df['similarity_score'] = similarities.cpu().numpy().flatten()

        df_sorted = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)

        df_sorted['match_percentage'] = (df_sorted['similarity_score'] * 100).round(1).astype(str) + '%'

        return df_sorted

    except Exception as e:
        logger.error(f"Error in ranking candidates: {str(e)}")
        return pd.DataFrame()

# Function to convert dataframe to Excel for download
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Ranked Candidates', index=False)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Routes
@app.route('/')
def index():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username_or_email = request.form['username'].lower() 
    password = request.form['password']
    users = load_users()

    user = None
    # Check both username and email match
    for u in users:
        if username_or_email == u.lower() or username_or_email == users[u].get('email', '').lower():
            user = u
            break

    if user and check_password(users[user]['password'], password):
        session['logged_in'] = True
        session['username'] = user 
        session['user_role'] = users[user].get('role', 'user')
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid username/email or password', 'danger')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('index'))
    return render_template('dashboard.html',
                         username=session['username'],
                         user_role=session['user_role'],
                         search_results=session.get('search_results'), 
                         ranked_results=session.get('ranked_results'))

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    country = request.form['location_country'] if 'location_country' in request.form else request.form['country']
    location = request.form['location']
    max_results = max(1, int(request.form['max_results']))
    
    # Define exclude patterns based on the query
    exclude_patterns = ["PA to", "Assistant to", "Personal Assistant", "Assistant", "Secretary to"]
    
    # If query contains CFO or Chief Financial Officer related terms, add specific exclusions
    if any(term.lower() in query.lower() for term in ["cfo", "chief financial officer", "finance"]):
        exclude_patterns.extend(["PA to CFO", "PA to Chief Financial", "Assistant to CFO", 
                                "Personal Assistant to CFO", "Secretary to the CFO"])
    
    results = search_employees_one_row_per_employee_dedup(query, country, location, max_results, exclude_patterns)
    if results.empty:
        flash('No candidates found matching your criteria.', 'warning')
    else:
        session['search_results'] = results.to_dict(orient='records')
        flash('Search successful!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/rank', methods=['POST'])
def rank():
    job_description = request.form['job_description']
    if 'search_results' not in session or not session['search_results']:
        flash('Please search for candidates first before ranking.', 'warning')
    else:
        df_employees = pd.DataFrame(session['search_results'])
        ranked_df = rank_candidates_semantic(df_employees, job_description)
        if ranked_df.empty:
            flash('Error occurred during ranking. Please try again.', 'danger')
        else:
            session['ranked_results'] = ranked_df.to_dict(orient='records')
            flash('Candidates ranked successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/download')
def download():
    if 'ranked_results' in session and session['ranked_results']:
        ranked_df = pd.DataFrame(session['ranked_results'])
        excel_data = to_excel(ranked_df)
        return send_file(io.BytesIO(excel_data), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='ranked_candidates.xlsx')
    else:
        flash('No ranked results available for download.', 'warning')
        return redirect(url_for('dashboard'))

@app.route('/download_search')
def download_search():
    if 'search_results' in session and session['search_results']:
        search_df = pd.DataFrame(session['search_results'])
        excel_data = to_excel(search_df)
        return send_file(io.BytesIO(excel_data), 
                       mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                       as_attachment=True, 
                       download_name='search_results.xlsx')
    else:
        flash('No search results available for download.', 'warning')
        return redirect(url_for('dashboard'))


# User Management Routes
@app.route('/manage_users')
def manage_users():
    if 'logged_in' not in session or session['user_role'] != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    users = load_users()
    return render_template('manage_users.html', 
                         users=users,
                         username=session['username'],
                         user_role=session['user_role'])

@app.route('/add_user', methods=['POST'])
def add_user():
    if 'logged_in' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    username = request.form['username'].strip()
    email = request.form['email'].strip()
    password = request.form['password'].strip()
    role = request.form['role'].strip()

    users = load_users()
    
    if not all([username, email, password, role]):
        flash('All fields are required', 'danger')
        return redirect(url_for('manage_users'))
    
    if username in users:
        flash('Username already exists', 'danger')
        return redirect(url_for('manage_users'))

    users[username] = {
        'password': make_hashed_password(password),
        'email': email,
        'created_at': datetime.now(),
        'role': role
    }
    
    save_users(users)
    flash('User added successfully', 'success')
    return redirect(url_for('manage_users'))

@app.route('/delete_user/<username>')
def delete_user(username):
    if 'logged_in' not in session or session['user_role'] != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    if username == 'admin':
        flash('Cannot delete primary admin', 'warning')
        return redirect(url_for('manage_users'))
    
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        flash('User deleted successfully', 'success')
    else:
        flash('User not found', 'danger')
    
    return redirect(url_for('manage_users'))

if __name__ == '__main__':
    app.run(debug=True)