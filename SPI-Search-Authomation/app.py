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

def search_employees_one_row_per_employee_dedup(
    query,
    country_filter=None,
    location_filter=None,
    company_filter=None,
    university_filter=None,
    industry_filter=None,
    skills_filter=None,
    certifications_filter=None,
    languages_filter=None,
    max_to_fetch=None
):
    """
    Search employees by:
      - 'query' (e.g. 'CEO', 'CEO OR CFO', etc.)
      - Optional filters:
            country_filter (e.g. 'South Africa'),
            location_filter (e.g. 'Johannesburg, Gauteng, South Africa'),
            company_filter (search in company names),
            university_filter (search in university names),
            industry_filter (search in the top-level industry field),
            skills_filter (search in skills),
            certifications_filter (search in certifications),
            languages_filter (search in languages),
            projects_filter (provided for consistency but not used in the search query).

    In the final DataFrame (one row per employee):
      - Keeps: ID, Name, Headline/Title, Location, Country, URL, Canonical_URL, Industry,
               Experience Count, Summary.
      - Includes: deduplicated Experiences (with duration), Educations, Skills, Certifications,
                  Languages, and Projects.
    """
    # Build the list of must clauses.
    must_clauses = []

    # Base clause: search in experience title.
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

    # Additional filter: Company Name (in experience collection)
    if company_filter:
        must_clauses.append({
            "nested": {
                "path": "member_experience_collection",
                "query": {
                    "match_phrase": {
                        "member_experience_collection.company_name": company_filter
                    }
                }
            }
        })

    # Additional filter: University Name (in education collection)
    if university_filter:
        must_clauses.append({
            "nested": {
                "path": "member_education_collection",
                "query": {
                    "match_phrase": {
                        "member_education_collection.title": university_filter
                    }
                }
            }
        })

    # Additional filter: Industry (top-level field)
    if industry_filter:
        must_clauses.append({
            "match_phrase": {
                "industry": industry_filter
            }
        })

    # Additional filter: Skills (in skills collection)
    if skills_filter:
        must_clauses.append({
            "nested": {
                "path": "member_skills_collection",
                "query": {
                    "match_phrase": {
                        "member_skills_collection.member_skill_list.skill": skills_filter
                    }
                }
            }
        })

    # Additional filter: Certifications (in certifications collection)
    if certifications_filter:
        must_clauses.append({
            "nested": {
                "path": "member_certifications_collection",
                "query": {
                    "match_phrase": {
                        "member_certifications_collection.name": certifications_filter
                    }
                }
            }
        })

    # Additional filter: Languages (in languages collection)
    if languages_filter:
        # Convert the search term to lower case so that "English" matches stored "english"
        must_clauses.append({
            "nested": {
                "path": "member_languages_collection",
                "query": {
                    "match_phrase": {
                        "member_languages_collection.member_language_list.language": languages_filter.lower()
                    }
                }
            }
        })

    # Exclude patterns in titles
    exclude_patterns = ["PA to", "Assistant to", "Personal Assistant", "EA to","Executive Assistant to","CFO Designate","CEO Designate"]
    must_not_clauses = [
        {
            "nested": {
                "path": "member_experience_collection",
                "query": {
                    "query_string": {
                        "query": f"member_experience_collection.title:({pattern})",
                        "default_operator": "or"
                    }
                }
            }
        }
        for pattern in exclude_patterns
    ]

    # Build the complete payload with country and location filters added.
    payload = {
        "query": {
            "bool": {
                "must": must_clauses,
                "must_not": must_not_clauses
            }
        }
    }

    if country_filter:
        payload["query"]["bool"]["must"].append({
            "term": {
                "country": country_filter
            }
        })

    if location_filter:
        payload["query"]["bool"]["must"].append({
            "match_phrase": {
                "location": location_filter
            }
        })

    # Uncomment for debugging:
    # print(json.dumps(payload, indent=2))

    # Send the search request.
    search_url = "https://api.coresignal.com/cdapi/v1/professional_network/employee/search/es_dsl"
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJFZERTQSIsImtpZCI6IjYzOGY5Y2YyLTUyM2UtOGJmMC0zZmFlLTEyY2UwNTUzOTQ1YiJ9.eyJhdWQiOiJzdHVkZW50LnVqLmFjLnphIiwiZXhwIjoxNzczMjc2NTkzLCJpYXQiOjE3NDE3MTk2NDEsImlzcyI6Imh0dHBzOi8vb3BzLmNvcmVzaWduYWwuY29tOjgzMDAvdjEvaWRlbnRpdHkvb2lkYyIsIm5hbWVzcGFjZSI6InJvb3QiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJzdHVkZW50LnVqLmFjLnphIiwic3ViIjoiOTc4OGQ4OTYtMjcwYy01ODY4LTE2NDItOTFhYmQ5NDBhMDg2IiwidXNlcmluZm8iOnsic2NvcGVzIjoiY2RhcGkifX0.GYI_XfOwh_DiuBMu9q_JRL39v4bOgJixOWIxPG0ZujADWVFtQQKO1tNJ71ig-ncoRJJE7R6z0WbG4Bxjs_qkDw'
    }
    resp = requests.post(search_url, headers=headers, json=payload)
    resp.raise_for_status()
    employee_ids = resp.json()

    if not isinstance(employee_ids, list):
        print("Unexpected structure in search response.")
        return pd.DataFrame()

    # Collect data for each employee ID.
    rows = []
    for emp_id in employee_ids[:max_to_fetch]:
        collect_url = f"https://api.coresignal.com/cdapi/v1/professional_network/employee/collect/{emp_id}"
        r = requests.get(collect_url, headers=headers)
        r.raise_for_status()
        employee = r.json()

        # Basic fields
        id_val = employee.get("id")
        name_val = employee.get("name")
        headline_val = employee.get("title")
        location_val = employee.get("location")
        country_val = employee.get("country")
        url_val = employee.get("url")
        canonical_url = employee.get("canonical_url")
        industry_val = employee.get("industry")
        experience_count_val = employee.get("experience_count")
        summary_val = employee.get("summary")

        # ----- EXPERIENCE (deduplicate) -----
        raw_exps = employee.get("member_experience_collection", [])
        unique_exps = []
        seen_exps = set()
        for exp in raw_exps:
            key = (
                exp.get("title", "N/A"),
                exp.get("company_name", "N/A"),
                exp.get("date_from", "N/A"),
                exp.get("date_to", "N/A")
            )
            if key not in seen_exps:
                seen_exps.add(key)
                unique_exps.append(exp)
        experiences_str = "\n".join(
            f"Role: {exp.get('title','N/A')} | Company: {exp.get('company_name','N/A')} | From: {exp.get('date_from','N/A')} | To: {exp.get('date_to','N/A')} | Duration: {exp.get('duration','N/A')}"
            for exp in unique_exps
        )

        # ----- EDUCATION (deduplicate) -----
        raw_edu = employee.get("member_education_collection", [])
        unique_edu = []
        seen_edu = set()
        for edu in raw_edu:
            key = (
                edu.get("title", "N/A"),
                edu.get("subtitle", "N/A"),
                edu.get("date_from", "N/A"),
                edu.get("date_to", "N/A")
            )
            if key not in seen_edu:
                seen_edu.add(key)
                unique_edu.append(edu)
        educations_str = "\n".join(
            f"Institution: {edu.get('title','N/A')} | Degree: {edu.get('subtitle','N/A')} | From: {edu.get('date_from','N/A')} | To: {edu.get('date_to','N/A')}"
            for edu in unique_edu
        )

        # ----- SKILLS (deduplicate) -----
        raw_skills = employee.get("member_skills_collection", [])
        seen_skills = set()
        for skill_entry in raw_skills:
            skill_name = skill_entry.get("member_skill_list", {}).get("skill", "N/A")
            seen_skills.add(skill_name)
        skills_str = ", ".join(seen_skills) if seen_skills else ""

        # ----- CERTIFICATIONS (deduplicate) -----
        raw_certifications = employee.get("member_certifications_collection", [])
        seen_certs = set()
        for cert in raw_certifications:
            cert_name = cert.get("name", "N/A")
            seen_certs.add(cert_name)
        certifications_str = ", ".join(seen_certs) if seen_certs else ""

        # ----- LANGUAGES (deduplicate) -----
        raw_languages = employee.get("member_languages_collection", [])
        seen_langs = set()
        for lang in raw_languages:
            language_name = lang.get("member_language_list", {}).get("language", "N/A")
            seen_langs.add(language_name)
        languages_str = ", ".join(seen_langs) if seen_langs else ""

        # ----- PROJECTS (deduplicate) -----
        raw_projects = employee.get("member_projects_collection", [])
        seen_projects = set()
        for proj in raw_projects:
            proj_name = proj.get("name", "N/A")
            seen_projects.add(proj_name)
        projects_str = ", ".join([str(x) for x in seen_projects if x is not None]) if seen_projects else ""

        # Build the final row dictionary.
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
            "Skills": skills_str,
            "Certifications": certifications_str,
            "Languages": languages_str,
            "Projects": projects_str
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# Ranking model
def build_user_text(row, text_columns: List[str]) -> str:
    """
    Combine relevant text fields into a single string for semantic comparison.
    Handles both string and list-type columns.
    
    Args:
        row: DataFrame row containing user information
        text_columns: List of columns to include in combined text
        
    Returns:
        Combined text string
    """
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
    """
    Clean and normalize text input by:
    - Removing emojis and special characters
    - Removing extra whitespace
    - Converting to lowercase
    """
    # Remove emojis using Unicode range patterns
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # CJK symbols
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
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special characters and punctuation (keep alphanumeric and whitespace)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase and clean whitespace
    text = text.lower()
    text = ' '.join(text.strip().split())
    return text

def rank_candidates_semantic(
    df_employees: pd.DataFrame,
    job_description: str,
    text_columns: Optional[List[str]] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Rank candidates based on semantic similarity to job description.
    """
    try:
        logger.info("Starting candidate ranking process...")
        
        # Create working copy to avoid modifying original dataframe
        df = df_employees.copy()
        
        # Set columns for corpus
        if text_columns is None:
            text_columns = ['Summary', 'Experiences', 'Educations', 
                           'Headline/Title', 'Industry', 'Skills'
                           'Certifications','Projects']
            logger.debug(f"Using default text columns: {text_columns}")
        else:
            logger.debug(f"Using custom text columns: {text_columns}")

        # 1) Create combined text for each user
        logger.info("Combining candidate text fields...")
        df['combined_text'] = df.apply(
            lambda x: build_user_text(x, text_columns), 
            axis=1
        )
        logger.info(f"Processed {len(df)} candidate profiles")

        # Handle empty texts to avoid encoding issues
        logger.info("Filtering empty candidate texts...")
        initial_count = len(df)
        df['combined_text'] = df['combined_text'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['combined_text']).reset_index(drop=True)
        filtered_count = len(df)
        logger.info(f"Removed {initial_count - filtered_count} empty profiles, {filtered_count} remaining")

        if df.empty:
            logger.warning("No valid candidate texts found after preprocessing")
            return pd.DataFrame()

        # 2) Initialize sentence transformer model
        logger.info(f"Initializing sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # 3) Preprocess and embed job description
        logger.info("Preprocessing job description...")
        clean_jd = preprocess_text(job_description)
        logger.debug(f"Job description length: {len(clean_jd.split())} words")
        
        logger.info("Encoding job description...")
        job_embedding = model.encode(clean_jd, convert_to_tensor=True)
        logger.debug(f"Job embedding shape: {job_embedding.shape}")

        # 4) Embed candidate texts in batches
        logger.info("Preprocessing candidate texts...")
        user_texts = df['combined_text'].apply(preprocess_text).tolist()
        logger.debug(f"First candidate text preview: {user_texts[0][:200]}...")
        
        logger.info(f"Encoding candidate texts in batches of {batch_size}...")
        user_embeddings = model.encode(
            user_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=True
        )
        logger.info(f"Successfully encoded {len(user_texts)} candidate texts")
        logger.debug(f"Embeddings matrix shape: {user_embeddings.shape}")

        # 5) Calculate cosine similarities
        logger.info("Calculating cosine similarities...")
        similarities = util.cos_sim(job_embedding, user_embeddings)
        df['similarity_score'] = similarities.cpu().numpy().flatten()
        
        # Calculate score statistics
        min_score = df['similarity_score'].min()
        max_score = df['similarity_score'].max()
        logger.info(f"Similarity scores range: {min_score:.3f} - {max_score:.3f}")
        logger.debug(f"Score distribution:\n{df['similarity_score'].describe()}")

        # 6) Sort and return results
        logger.info("Sorting candidates by similarity score...")
        df_sorted = df.sort_values(by='similarity_score', ascending=False)\
                      .reset_index(drop=True)
        
        logger.info(f"Top candidate score: {df_sorted.iloc[0]['similarity_score']:.3f}")
        logger.info("Ranking process completed successfully")
        
        return df_sorted

    except Exception as e:
        logger.error(f"Error in ranking candidates: {str(e)}")
        raise

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
    # Existing fields
    query = request.form['query']
    country = request.form.get('country')
    location = request.form.get('location')
    max_results = max(1, int(request.form.get('max_results', 50)))
    
    # New filters
    company_filter = request.form.get('company_filter')
    university_filter = request.form.get('university_filter')
    industry_filter = request.form.get('industry_filter')
    skills_filter = request.form.get('skills_filter')
    certifications_filter = request.form.get('certifications_filter')
    languages_filter = request.form.get('languages_filter')
    
    # Define exclude patterns (existing logic)
    exclude_patterns = ["PA to", "Assistant to", "Personal Assistant", "Assistant", "Secretary to"]
    if any(term.lower() in query.lower() for term in ["cfo", "chief financial officer", "finance"]):
        exclude_patterns.extend(["PA to CFO", "PA to Chief Financial", "Assistant to CFO", 
                                "Personal Assistant to CFO", "Secretary to the CFO",
                                "EA to","Executive Assistant to","CFO Designate","CEO Designate"])
    
    # Updated search function call with all filters
    results = search_employees_one_row_per_employee_dedup(
        query,
        country_filter=country,
        location_filter=location,
        company_filter=company_filter,
        university_filter=university_filter,
        industry_filter=industry_filter,
        skills_filter=skills_filter,
        certifications_filter=certifications_filter,
        languages_filter=languages_filter,
        max_to_fetch=max_results
    )
    
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