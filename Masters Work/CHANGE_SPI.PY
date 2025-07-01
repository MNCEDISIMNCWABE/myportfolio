
import streamlit as st
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
import json
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from datetime import date
import calendar

# Set up logging and ignore warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoCreditsError(Exception):
    """Custom exception raised when the Proxycurl API indicates no credits."""
    pass

# User authentication functions
def make_hashed_password(password):
    """Create a hashed version of the password."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(stored_password, input_password):
    """Check if the input password matches the stored password."""
    return stored_password == make_hashed_password(input_password)

SHEET_URL = "https://docs.google.com/spreadsheets/d/1VRkbyccd0dRJdkqzKL8g6DS4-AoFwbatQefMrvyJnmc/edit?gid=0#gid=0"

@st.cache_resource(show_spinner="Initializing Application...")
def get_google_sheets_client(sheet_url):
    """Authenticate and return a Google Sheets client using Streamlit secrets."""
    try:
        creds_json = st.secrets["GOOGLE_CREDS_FILE"]
        if not creds_json:
            raise ValueError("Google credentials not found in Streamlit secrets.")
        
        creds_dict = json.loads(creds_json)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open_by_url(sheet_url).sheet1
    except Exception as e:
        logger.error(f"Error initializing Google Sheets client: {str(e)}")
        raise

def init_db(credentials_file=None):
    sheet = get_google_sheets_client(SHEET_URL)
    try:
        if not sheet.get_all_records():
            sheet.append_row(["id", "username", "email", "password", "role", "created_at"])
            logger.info("Google Sheet initialized with headers.")
        
        users = load_users(sheet)
        if "admin" not in users:
            save_users("admin", "SPI123@_", "admin@example.com", "admin", sheet)
            logger.info("Default admin user inserted.")
    except Exception as e:
        logger.error(f"Error initializing Google Sheet: {str(e)}")

# Load users from Google Sheet
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_users(_sheet): 
    users = {}
    try:
        records = _sheet.get_all_records()
        for row in records:
            try:
                created_at = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
            except:
                created_at = datetime.now()  # fallback if date parsing fails
            
            users[row["username"]] = {
                "id": row["id"],
                "email": row["email"],
                "password": row["password"],
                "role": row["role"],
                "created_at": created_at,
            }
        logger.info(f"Loaded {len(users)} users from Google Sheet")
    except Exception as e:
        logger.error(f"Error loading users from Google Sheet: {str(e)}")
    return users

def save_users(username, password, email, role, sheet):
    try:
        users = load_users(sheet) 
        if username in users:
            cell = sheet.find(username)
            sheet.update_cell(cell.row, 3, email)
            sheet.update_cell(cell.row, 4, make_hashed_password(password))
            sheet.update_cell(cell.row, 5, role)
        else:
            next_id = len(users) + 1
            sheet.append_row([next_id, username, email, make_hashed_password(password), role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        logger.info(f"User '{username}' saved successfully.")
    except Exception as e:
        logger.error(f"Error saving user to Google Sheet: {str(e)}")


def delete_user(username, sheet):
    try:
        cell = sheet.find(username)  # Find the row with the username
        sheet.delete_rows(cell.row)  # Corrected method
        logger.info(f"User '{username}' deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting user from Google Sheet: {str(e)}")


def search_employees_one_row_per_employee_dedup(
    query,
    country_filter=None,
    location_filter=None,
    university_filter=None,
    province_filter=None,
    skills_filter=None,
    industry_filter=None,
    company_filter=None,
    languages_filter=None,
    max_to_fetch=None
) -> List[str]:
    """
    Calls the Proxycurl /search/person endpoint with the given parameters
    and returns a list of LinkedIn URLs from the response.
    Raises:
        NoCreditsError: If the API returns 403 (insufficient credits).
        ValueError: If no results or no linkedin_profile_url is found.
    """
    headers = {'Authorization': f'Bearer E89sEU4Cw2jwkjoPjDHkfQ'}
    search_endpoint = 'https://enrichlayer.com/api/v2/search/person'
    
    search_params = {
        'past_role_title': query,
        'country': country_filter,
        'city': location_filter,
        'education_school_name': university_filter,
        'region': province_filter,
        'skills': skills_filter,
        'industries': industry_filter,
        'past_company_name': company_filter,
        'languages': languages_filter,
        'page_size': max_to_fetch,
        'use_cache': 'if-present'
    }
    
    resp = requests.get(search_endpoint, headers=headers, params=search_params)
    
    # Detect out-of-credits scenario
    if resp.status_code == 403:
        raise NoCreditsError("Your current Proxycurl credit balance is insufficient to run this search. Please top up your credits and try again.")
    
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError("No results returned from Search Person endpoint.")
    
    linkedin_urls = [result.get("linkedin_profile_url") for result in results if result.get("linkedin_profile_url")]
    if not linkedin_urls:
        raise ValueError("No linkedin_profile_url found in the search results.")
    
    return linkedin_urls


def enrich_profile(linkedin_url: str) -> dict:
    """
    Calls the Proxycurl /api/v2/linkedin endpoint to fetch detailed
    LinkedIn profile data given a LinkedIn URL.
    Returns the JSON (dictionary) with the user's profile data.
    """
    headers = {'Authorization': f'Bearer E89sEU4Cw2jwkjoPjDHkfQ'}
    api_endpoint = 'https://enrichlayer.com/api/v2/profile'
    
    params = {
        'linkedin_profile_url': linkedin_url,
        'personal_contact_number': 'include',
        'personal_email': 'include',
        'skills': 'include',
        'use_cache': 'if-present',
        'fallback_to_cache': 'on-error',
    }
    
    resp = requests.get(api_endpoint, headers=headers, params=params)
    data = resp.json()
    
    return data


def flatten_profile_data(profile_data: dict, linkedin_url: str) -> dict:
    """
    Given the enriched LinkedIn profile JSON and its LinkedIn URL,
    return a single flat dictionary with pipe-delimited strings for
    experiences, education, skills, certifications, etc.
    Each subsequent line in experiences/education is prefixed with "| ".
    """
    def format_date(date_dict):
        """Return 'MonthName YYYY' or 'None' if missing/invalid."""
        if not date_dict or "year" not in date_dict or "month" not in date_dict:
            return "None"
        return f"{calendar.month_name[date_dict['month']]} {date_dict['year']}"

    def compute_duration_months(start_dict, end_dict=None):
        """Compute duration in months between dates."""
        if not start_dict or "year" not in start_dict or "month" not in start_dict:
            return None
        today = date(2025, 3, 25)  # Fixed date for consistency
        start = start_dict["year"] * 12 + (start_dict["month"] - 1)
        end = (end_dict["year"] * 12 + (end_dict["month"] - 1) if end_dict 
               else today.year * 12 + (today.month - 1))
        return end - start

    def format_experience(exp):
        """Format a single experience entry."""
        title = exp.get("title", "")
        company = exp.get("company", "")
        start = format_date(exp.get("starts_at"))
        end = format_date(exp.get("ends_at"))
        duration = compute_duration_months(exp.get("starts_at"), exp.get("ends_at"))
        duration_str = f"{duration} months" if duration else "Unknown"
        return (f"Role: {title} | Company: {company} | From: {start} | "
                f"To: {end} | Duration: {duration_str} |")

    def format_education(edu):
        """Format a single education entry."""
        school = edu.get("school", "")
        degree = edu.get("degree_name", "")
        start = format_date(edu.get("starts_at"))
        end = format_date(edu.get("ends_at"))
        return (f"Institution: {school} | Degree: {degree} | "
                f"From: {start} | To: {end} |")

    def format_certification(cert):
        """Format a single certification entry."""
        name = cert.get("name", "")
        authority = cert.get("authority", "")
        start = format_date(cert.get("starts_at"))
        end = format_date(cert.get("ends_at")) if cert.get("ends_at") else "Present"
        return (f"Name: {name} | Authority: {authority} | "
                f"From: {start} | To: {end} |")

    # Handle languages field which might be a string or list of dicts
    languages = profile_data.get("languages", [])
    if isinstance(languages, str):
        languages_str = languages
    elif isinstance(languages, list):
        try:
            languages_str = " | ".join(
                lang.get("name", "") if isinstance(lang, dict) else str(lang)
                for lang in languages
            )
        except Exception as e:
            logger.warning(f"Error processing languages: {str(e)}")
            languages_str = ""
    else:
        languages_str = str(languages)

    # Extract top-level fields
    record = {
        "full_name": profile_data.get("full_name", ""),
        "country": profile_data.get("country", ""),
        "country_full_name": profile_data.get("country_full_name", ""),
        "province": profile_data.get("state", ""),
        "city": profile_data.get("city", ""),
        "personal_emails": ", ".join(profile_data.get("personal_emails", [])),
        "personal_numbers": ", ".join(profile_data.get("personal_numbers", [])),
        "URL": linkedin_url,
        "gender": profile_data.get("gender"),
        "headline": profile_data.get("headline", ""),
        "summary": profile_data.get("summary", ""),
        "industry": profile_data.get("industry"),
        "experiences": "\n".join([format_experience(exp) for exp in profile_data.get("experiences", [])]),
        "education": "\n".join([format_education(edu) for edu in profile_data.get("education", [])]),
        "skills": " | ".join(profile_data.get("skills", [])),
        "certifications": "\n".join([format_certification(cert) for cert in profile_data.get("certifications", [])]),
        "languages": languages_str
    }
    
    return record

def run_script(
    query: str,
    country_filter=None,
    location_filter=None,
    university_filter=None,
    province_filter=None,
    skills_filter=None,
    industry_filter=None,
    company_filter=None,
    languages_filter=None,
    max_to_fetch=None
) -> pd.DataFrame:
    """
    Executes the search and enrichment process. Returns:
      - A DataFrame with candidate data if successful,
      - An empty DataFrame if no candidates are found,
      - A DataFrame with a 'credits_error' column if credits are insufficient.
    """
    try:
        linkedin_urls = search_employees_one_row_per_employee_dedup(
            query,
            country_filter=country_filter,
            location_filter=location_filter,
            university_filter=university_filter,
            province_filter=province_filter,
            skills_filter=skills_filter,
            industry_filter=industry_filter,
            company_filter=company_filter,
            languages_filter=languages_filter,
            max_to_fetch=max_to_fetch
        )
    except NoCreditsError:
        return pd.DataFrame({"credits_error": [True]})
    except ValueError:
        return pd.DataFrame()
    
    records = []
    for url in linkedin_urls:
        profile_json = enrich_profile(linkedin_url=url)
        record = flatten_profile_data(profile_data=profile_json, linkedin_url=url)
        records.append(record)
    
    df_employees = pd.DataFrame(records)
    return df_employees

# Ranking functions
def build_user_text(row, text_columns: List[str]) -> str:
    """
    Combine relevant text fields into a single string for semantic comparison.
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
    Clean and normalize text input.
    """
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

def rank_candidates_semantic(
    df_employees: pd.DataFrame,
    job_description: str,
    text_columns: Optional[List[str]] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> pd.DataFrame:
    try:
        logger.info("Starting candidate ranking process...")
        df = df_employees.copy()
        
        if text_columns is None:
            text_columns = [
                'summary', 'experiences', 'education', 'headline',
                'industry', 'skills', 'certifications'
            ]
            
            logger.debug(f"Using default text columns: {text_columns}")
        else:
            logger.debug(f"Using custom text columns: {text_columns}")

        logger.info("Combining candidate text fields...")
        df['combined_text'] = df.apply(lambda x: build_user_text(x, text_columns), axis=1)
        logger.info(f"Processed {len(df)} candidate profiles")

        logger.info("Filtering empty candidate texts...")
        initial_count = len(df)
        df['combined_text'] = df['combined_text'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['combined_text']).reset_index(drop=True)
        filtered_count = len(df)
        logger.info(f"Removed {initial_count - filtered_count} empty profiles, {filtered_count} remaining")

        if df.empty:
            logger.warning("No valid candidate texts found after preprocessing")
            return pd.DataFrame()

        logger.info(f"Initializing sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info("Preprocessing job description...")
        clean_jd = preprocess_text(job_description)
        logger.debug(f"Job description length: {len(clean_jd.split())} words")
        
        logger.info("Encoding job description...")
        job_embedding = model.encode(clean_jd, convert_to_tensor=True)
        logger.debug(f"Job embedding shape: {job_embedding.shape}")

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

        logger.info("Calculating cosine similarities...")
        similarities = util.cos_sim(job_embedding, user_embeddings)
        df['similarity_score'] = similarities.cpu().numpy().flatten()
        df['match_percentage'] = (df['similarity_score'] * 100).round(2).astype(str) + '%'
        
        min_score = df['similarity_score'].min()
        max_score = df['similarity_score'].max()
        logger.info(f"Similarity scores range: {min_score:.3f} - {max_score:.3f}")
        logger.debug(f"Score distribution:\n{df['similarity_score'].describe()}")

        logger.info("Sorting candidates by similarity score...")
        df_sorted = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
        df_sorted = df_sorted.drop('combined_text', axis=1)

        logger.info(f"Top candidate score: {df_sorted.iloc[0]['similarity_score']:.3f}")
        logger.info("Ranking process completed successfully")
        return df_sorted

    except Exception as e:
        logger.error(f"Error in ranking candidates: {str(e)}")
        raise

# Cache the model to avoid reloading
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# Function to convert dataframe to Excel for download
def to_excel(df):
    """Robust Excel export that handles all data types and empty values"""
    output = io.BytesIO()
    
    # Create a clean copy of the DataFrame
    export_df = df.copy()
    
    # Convert all columns to string and handle None/NaN values
    for col in export_df.columns:
        # Replace None/NaN with empty string
        export_df[col] = export_df[col].fillna('')
        # Convert all values to string
        export_df[col] = export_df[col].astype(str)
        # Remove any problematic characters
        export_df[col] = export_df[col].str.replace('\x00', '')  # Remove null bytes
    
    # Create Excel file with xlsxwriter
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, sheet_name='Candidates', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Candidates']
        for idx, col in enumerate(export_df.columns):
            max_len = max((
                export_df[col].astype(str).map(len).max(),
                len(str(col))
            )) + 1
            worksheet.set_column(idx, idx, min(max_len, 50))
    
    output.seek(0)
    return output.getvalue()

# Login management functions
def login_page():
    st.title("SPI Executive Search")
    login_tab, signup_tab, forgot_tab = st.tabs(["Login", "Sign Up", "Forgot Password"])

    sheet = get_google_sheets_client(SHEET_URL) 

    with login_tab:
        username_or_email = st.text_input("Username or Email", key="login_username_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            users = load_users(sheet) 
            
            if username_or_email in users:
                if check_password(users[username_or_email]['password'], password):
                    st.session_state.logged_in = True
                    st.session_state.username = username_or_email
                    st.session_state.user_role = users[username_or_email].get('role', 'user')
                    st.success(f"Welcome back, {username_or_email}!")
                    st.rerun()
                else:
                    st.error("Invalid password.")
            else:
                found = False
                for username, user_data in users.items():
                    if user_data.get('email') == username_or_email:
                        if check_password(user_data['password'], password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_role = user_data.get('role', 'user')
                            st.success(f"Welcome back, {username}!")
                            found = True
                            st.rerun()
                            break
                        else:
                            st.error("Invalid password.")
                            found = True
                            break
                
                if not found:
                    st.error("Invalid username/email or password.")
    
    with signup_tab:
        if st.session_state.get('user_role') == 'admin':
            new_username = st.text_input("New Username", key="new_username")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            email = st.text_input("Email", key="email")
            
            if st.button("Sign Up"):
                users = load_users(sheet)  # Pass the sheet object
                # Check if username already exists
                if new_username in users:
                    st.error("Username already exists")
                # Check if email already exists
                elif any(user_data.get('email') == email for user_data in users.values()):
                    st.error("Email already in use")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_username or not new_password:
                    st.error("Username and password cannot be empty")
                elif not email:
                    st.error("Email cannot be empty")
                else:
                    # Save the new user to the database
                    save_users(new_username, new_password, email, 'user', sheet)  # Added sheet parameter
                    st.success("Account created successfully! You can now login.")
        else:
            st.info("User registration is only managed by administrators. Please contact your administrator for access.")

    with forgot_tab:
        st.subheader("Reset Password")
        username_or_email = st.text_input("Enter your username or email", key="reset_username_email")
        new_password = st.text_input("New Password", type="password", key="reset_new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
        
        if st.button("Reset Password", key="reset_password_button"):
            users = load_users(sheet)  # Pass the sheet object
            user_found = False
            username = None
            
            # Check if the input is a username
            if username_or_email in users:
                username = username_or_email
                user_found = True
            else:
                # Check if the input is an email
                for u, data in users.items():
                    if data.get('email', '').lower() == username_or_email.lower():
                        username = u
                        user_found = True
                        break
            
            if not user_found:
                st.error("No account found with the provided username or email.")
            elif new_password != confirm_password:
                st.error("New passwords do not match. Please try again.")
            else:
                # Update the password
                hashed_password = make_hashed_password(new_password)  # Hash the new password
                save_users(username, new_password, users[username]['email'], users[username]['role'], sheet)
                
                # Clear the cache to ensure fresh data is loaded
                st.cache_data.clear()
                st.session_state.user_cache_clear = True
                
                st.success("Password has been reset successfully. You can now login with your new password.")
                time.sleep(2)  # Show message for 2 seconds
                st.rerun()


def logout():
    if st.sidebar.button("Logout"):
        for key in ['logged_in', 'username', 'user_role']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


def admin_dashboard():
    st.title("Admin Dashboard - User Management")
    sheet = get_google_sheets_client(SHEET_URL)
    
    # Clear the cache before loading users to ensure we get fresh data
    if 'user_cache_clear' in st.session_state:
        st.cache_data.clear()
        del st.session_state.user_cache_clear
    
    users = load_users(sheet)  # Pass the sheet object
    user_df = pd.DataFrame([
        {
            'Username': username,
            'Email': data['email'],
            'Created At': data['created_at'].strftime('%Y-%m-%d %H:%M:%S'),  
            'Role': data.get('role', 'user')
        }
        for username, data in users.items()
    ])
    st.dataframe(user_df)
    
    st.subheader("Add New User")
    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("Username", key="admin_new_username")
        new_password = st.text_input("Password", type="password", key="admin_new_password")
    with col2:
        email = st.text_input("Email", key="admin_email")
        role = st.selectbox("Role", ["user", "admin"], key="admin_role")
    
    if st.button("Add User"):
        if new_username in users:
            st.error("Username already exists")
        elif not new_username or not new_password:
            st.error("Username and password cannot be empty")
        else:
            save_users(new_username, new_password, email, role, sheet)
            # Set flag to clear cache on next run
            st.session_state.user_cache_clear = True
            st.success(f"User '{new_username}' added successfully")
            st.rerun()
    
    st.subheader("Delete User")
    username_to_delete = st.selectbox("Select User to Delete", list(users.keys()))
    if st.button("Delete User") and username_to_delete:
        if username_to_delete == st.session_state.username:
            st.error("You cannot delete your own account while logged in!")
        else:
            delete_user(username_to_delete, sheet)
            # Set flag to clear cache on next run
            st.session_state.user_cache_clear = True
            st.success(f"User '{username_to_delete}' deleted successfully")
            st.rerun()


def main():
    st.set_page_config(page_title="Candidate Search & Match", layout="wide")

     # Initialize Google Sheets:
    sheet = get_google_sheets_client(SHEET_URL)
    if 'db_initialized' not in st.session_state:
        init_db()
        st.session_state.db_initialized = True
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'ranked_results' not in st.session_state:
        st.session_state.ranked_results = None
    
    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        st.sidebar.write(f"Role: **{st.session_state.user_role}**")
        logout()
        if st.session_state.user_role == 'admin':
            pages = ["Candidate Search", "Admin Dashboard"]
            selected_page = st.sidebar.selectbox("Navigation", pages)
            if selected_page == "Admin Dashboard":
                admin_dashboard()
                return

    if not st.session_state.logged_in:
        login_page()
        return

    st.title("Candidate Search & Match")
    st.markdown("Find and rank the best candidates for a job position")
    tab1, tab2 = st.tabs(["Search Candidates", "Ranked Results"])
    
    with tab1:
        st.header("Search for Candidates")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Search Criteria")
            search_query = st.text_input("Job Title/Position", placeholder="e.g. Chief Financial Officer OR CFO")
            
            # Location filters
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                country = st.text_input("Country", placeholder="e.g. ZA")
            with loc_col2:
                location = st.text_input("City", placeholder="e.g. Johannesburg")
            
            # Province and Company filters
            prov_col1, prov_col2 = st.columns(2)
            with prov_col1:
                province = st.text_input("Province", placeholder="e.g. Gauteng")
            with prov_col2:
                company_filter = st.text_input("Company Name", placeholder="e.g. PWC")
            
            # Education and Industry filters
            edu_col1, edu_col2 = st.columns(2)
            with edu_col1:
                university_filter = st.text_input("University Name", placeholder="e.g. University of Cape Town")
            with edu_col2:
                industry_filter = st.text_input("Industry", placeholder="e.g. Accounting")
            
            # Skills and Certifications filters
            skills_col1, skills_col2 = st.columns(2)
            with skills_col1:
                skills_filter = st.text_input("Skills", placeholder="e.g. business strategy, financial modeling")
            with skills_col2:
                certifications_filter = st.text_input("Certifications", placeholder="e.g. Assessor")
            
            # Languages filter (full width)
            languages_filter = st.text_input("Languages", placeholder="e.g. English")
            
            # Results and Search button
            slider_col, btn_col = st.columns([2, 1])
            with slider_col:
                max_results = st.slider("Maximum number of results", 1, 150, 15)
            with btn_col:
                st.write("")
                st.write("")
                search_button = st.button("Search Candidates")
            
            if search_button and search_query:
                with st.spinner("Searching for candidates..."):
                    st.session_state.ranked_results = None
                    results = run_script(
                        query=search_query,
                        country_filter=country if country else None,
                        location_filter=location if location else None,
                        province_filter=province if province else None,  
                        company_filter=company_filter if company_filter else None, 
                        university_filter=university_filter if university_filter else None,
                        industry_filter=industry_filter if industry_filter else None,
                        skills_filter=skills_filter if skills_filter else None,
                        languages_filter=languages_filter if languages_filter else None,
                        max_to_fetch=max_results
                    )

                    # Check for special "credits_error" marker
                    if not results.empty and "credits_error" in results.columns:
                        st.error("Your current Proxycurl credit balance is insufficient to run this search. Please top up your credits and try again.")
                    elif results.empty:
                        st.error("No candidates found matching your criteria.")
                    else:
                        st.session_state.search_results = results
                        st.success(f"Found {len(results)} candidates!")
        
        with col2:
            st.subheader("Job Description")
            st.markdown("Provide a detailed job description to rank candidates against:")
            job_description = st.text_area(
                "Enter job description", 
                height=250,
                placeholder="Paste detailed job description here to rank candidates by relevance..."
            )
            rank_button = st.button("Rank Candidates")
            
            if rank_button:
                if st.session_state.search_results is None or st.session_state.search_results.empty:
                    st.error("Please search for candidates first before ranking.")
                elif not job_description:
                    st.warning("Please provide a job description for ranking candidates.")
                else:
                    with st.spinner("Ranking candidates..."):
                        load_model()
                        ranked_df = rank_candidates_semantic(
                            df_employees=st.session_state.search_results,
                            job_description=job_description,
                            model_name='all-MiniLM-L6-v2'
                        )
                        if ranked_df.empty:
                            st.error("Error occurred during ranking. Please try again.")
                        else:
                            st.session_state.ranked_results = ranked_df
                            st.success("Candidates ranked successfully! View results in the 'Ranked Results' tab.")
        
        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
            st.subheader("Search Results")
            
            # Add download button for raw search results
            raw_export_df = st.session_state.search_results.copy()
            raw_excel_data = to_excel(raw_export_df)
            st.download_button(
                label="📥 Download Search Results (Excel)",
                data=raw_excel_data,
                file_name='search_results.xlsx',
                mime='application/vnd.ms-excel',
                key='raw_download'
            )
            
            for i, row in st.session_state.search_results.iterrows():
                with st.expander(f"{row['full_name']} - {row['headline']}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Location:** {row['city']}, {row['country']}")
                        st.markdown(f"**Province:** {row['province']}")
                        st.markdown(f"**Industry:** {row['industry']}")
                        st.markdown(f"**Profile URL:** [Link]({row['URL']})")
                        if pd.notnull(row['personal_emails']) and row['personal_emails']:
                            st.markdown(f"**Email:** {row['personal_emails']}")
                        if pd.notnull(row['personal_numbers']) and row['personal_numbers']:
                            st.markdown(f"**Phone:** {row['personal_numbers']}")
                    with col2:
                        if pd.notnull(row['summary']) and row['summary']:
                            st.markdown("**Summary:**")
                            st.markdown(row['summary'])
                        if pd.notnull(row['skills']) and row['skills']:
                            st.markdown("**Skills:**")
                            st.markdown(row['skills'])
                        if pd.notnull(row['languages']) and row['languages']:
                            st.markdown("**Languages:**")
                            st.markdown(row['languages'])
                    
                    if pd.notnull(row['experiences']) and row['experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    
                    if pd.notnull(row['education']) and row['education']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['education'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")
                    
                    if pd.notnull(row['certifications']) and row['certifications']:
                        st.markdown("---")
                        st.markdown("### Certifications")
                        certs = row['certifications'].split('\n')
                        for cert in certs:
                            st.markdown(f"- {cert}")
    with tab2:
        st.header("Ranked Candidates")
        if st.session_state.ranked_results is not None and not st.session_state.ranked_results.empty:
            export_columns = [
                'full_name', 'country', 'country_full_name', 'province', 'city', 
                'personal_emails', 'personal_numbers', 'URL', 'gender', 'headline', 
                'summary', 'industry', 'experiences', 'education', 'skills', 
                'certifications', 'languages', 'similarity_score'
            ]
            export_df = st.session_state.ranked_results[
                [col for col in export_columns if col in st.session_state.ranked_results.columns]
            ].copy()
            if 'similarity_score' in export_df.columns:
                export_df['similarity_score'] = export_df['similarity_score'] * 100
            excel_data = to_excel(export_df)
            st.download_button(
                label="📥 Download Ranked Candidates (Excel)",
                data=excel_data,
                file_name='ranked_candidates.xlsx',
                mime='application/vnd.ms-excel',
            )
            
            st.subheader("Match Results")
            top_candidates = st.session_state.ranked_results.head(10)
            chart_data = pd.DataFrame({
                'Candidate': top_candidates['full_name'],  # Changed from 'Name' to 'full_name'
                'Match Percentage': top_candidates['similarity_score'] * 100
            })
            st.bar_chart(chart_data.set_index('Candidate'))
            
            for i, row in st.session_state.ranked_results.iterrows():
                with st.expander(f"{row['full_name']} - {row['headline']} (Match: {row['match_percentage']})"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Match Score:** {row['match_percentage']}")
                        st.markdown(f"**Location:** {row['city']}")
                        st.markdown(f"**Country:** {row['country']}")
                        st.markdown(f"**Industry:** {row['industry']}")
                        st.markdown(f"**Profile URL:** [Link]({row['URL']})")
                    with col2:
                        if pd.notnull(row['summary']) and row['summary']:
                            st.markdown("**Summary:**")
                            st.markdown(row['summary'])
                        if pd.notnull(row['skills']) and row['skills']:
                            st.markdown("**Skills:**")
                            st.markdown(row['skills'])
                    if pd.notnull(row['experiences']) and row['experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    if pd.notnull(row['education']) and row['education']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['education'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")
        else:
            st.info("No ranked results available. Please search for candidates and rank them first.")

    st.markdown("---")
    st.markdown("""
    **How to use this application:**
    1. Enter a job title and optionally other parameters in the search boxes (use OR for multiple terms)
    2. Use slider to limit the number of results returned
    3. Click "Search Candidates" to find matching profiles
    4. Download the results as an Excel file
    5. Optionally, enter a job description to match candidates against
    6. Click "Rank Candidates" to sort candidates by relevance to the job description
    7. View detailed rankings in the "Ranked Results" tab
    8. Download the ranked candidates as an Excel file in the "Ranked Results" tab
    """)

if __name__ == "__main__":
    main()