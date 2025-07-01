import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
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

# User authentication functions
def make_hashed_password(password):
    """Create a hashed version of the password."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(stored_password, input_password):
    """Check if the input password matches the stored password."""
    return stored_password == make_hashed_password(input_password)

SHEET_URL = "https://docs.google.com/spreadsheets/d/15FFMP8aUFeeb3I43SAYDevJAw0m5v740Pn7rQCUwnUo/edit#gid=0"

@st.cache_resource(show_spinner="Initializing Application...")
# def get_google_sheets_client(sheet_url):
#     """Authenticate and return a Google Sheets client using local file."""
#     # Fall back to local file
#     local_creds_path = "/Users/mncedisimncwabe/Documents/Personal_Google_Service_Account/bright-arc-328707-3770ec13c1ec.json"
#     if not os.path.exists(local_creds_path):
#         raise ValueError(f"Local credentials file not found at {local_creds_path}.")

#     with open(local_creds_path, 'r') as f:
#         creds_dict = json.load(f)

#     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
#     creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
#     client = gspread.authorize(creds)
#     return client.open_by_url(sheet_url).sheet1

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
            save_users("admin", "Cand123@_", "admin@example.com", "admin", sheet)
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

def generate_dummy_data(num_records=100):
    """Generate realistic dummy data for candidates from various countries."""
    np.random.seed(42)

    # Define data for different countries
    countries_data = {
        "ZA": {
            "cities": ["Johannesburg", "Cape Town", "Durban", "Pretoria", "Port Elizabeth", "Bloemfontein", "East London"],
            "provinces": ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Free State", "North West", "Mpumalanga", "Limpopo", "Northern Cape"],
            "languages": ["English", "Zulu", "Xhosa", "Afrikaans", "Pedi", "Tswana", "Sotho", "Tsonga", "Swati", "Venda", "Ndebele"],
            "first_names": ["Stefan", "Thabo", "Anele", "Nomsa", "Sipho", "Zanele", "Lee", "Puleng", "Kgomotso", "Tshepo"],
            "last_names": ["Ross", "Nkosi", "Mthembu", "Khumalo", "Mokoena", "Mabaso", "Davids", "Xaba", "Ntuli", "Msimang"],
            "universities": ["University of Cape Town", "University of the Witwatersrand", "Stellenbosch University", "University of Pretoria", "University of Johannesburg"],
            "companies": ["MTN", "Vodacom", "Standard Bank", "FirstRand", "Sasol"]
        }
    }

    # Job titles and relevant skills and certifications
    job_titles = {
        "Data Scientist": {
            "skills": ["Python", "Machine Learning", "Data Analysis", "SQL", "R", "TensorFlow"],
            "certifications": ["Microsoft Certified Azure: Data Science Associate", "AWS Cloud Practitioner"]
        },
        "Software Engineer": {
            "skills": ["Java", "Python", "JavaScript", "C++", "Software Development", "Agile"],
            "certifications": ["Certified Software Engineer", "AWS Certification"]
        },
        "Product Manager": {
            "skills": ["Product Development", "Market Analysis", "Agile", "Scrum", "Leadership"],
            "certifications": ["Certified Product Manager", "Scrum Master Certification"]
        },
        "Financial Analyst": {
            "skills": ["Financial Modeling", "Data Analysis", "Excel", "Risk Management", "Accounting"],
            "certifications": ["Certified Financial Analyst", "Chartered Financial Analyst (CFA)"]
        },
        "Marketing Specialist": {
            "skills": ["Digital Marketing", "SEO", "Content Creation", "Social Media", "Market Research"],
            "certifications": ["Google Analytics Certification", "HubSpot Content Marketing Certification"]
        }
    }

    education_details = {
        "ZA": [
            {"institution": "University of Cape Town", "degree": "Bachelor of Science in Computer Science", "from": "2010", "to": "2014"},
            {"institution": "University of the Witwatersrand", "degree": "Master of Business Administration", "from": "2015", "to": "2017"},
            {"institution": "Stellenbosch University", "degree": "Bachelor of Commerce in Finance", "from": "2011", "to": "2014"},
            {"institution": "University of Pretoria", "degree": "Bachelor of Engineering", "from": "2012", "to": "2016"},
            {"institution": "University of Johannesburg", "degree": "Bachelor of Arts in Psychology", "from": "2010", "to": "2013"}
        ]
    }

    # Sample experience details without company names
    experience_details = {
        "Data Scientist": [
            {"role": "Data Scientist", "from": "April 2020", "to": "Present", "duration": "36 months"},
            {"role": "Senior Data Analyst", "from": "June 2017", "to": "March 2020", "duration": "34 months"},
        ],
        "Software Engineer": [
            {"role": "Software Engineer", "from": "January 2019", "to": "Present", "duration": "48 months"},
            {"role": "Junior Developer", "from": "May 2016", "to": "December 2018", "duration": "32 months"},
        ],
        "Product Manager": [
            {"role": "Product Manager", "from": "March 2018", "to": "Present", "duration": "40 months"},
            {"role": "Associate Product Manager", "from": "July 2015", "to": "February 2018", "duration": "30 months"},
        ],
        "Financial Analyst": [
            {"role": "Financial Analyst", "from": "September 2019", "to": "Present", "duration": "24 months"},
            {"role": "Junior Financial Analyst", "from": "January 2017", "to": "August 2019", "duration": "32 months"},
        ],
        "Marketing Specialist": [
            {"role": "Marketing Specialist", "from": "November 2018", "to": "Present", "duration": "38 months"},
            {"role": "Marketing Coordinator", "from": "February 2016", "to": "October 2018", "duration": "32 months"},
        ]
    }

    # Sample summary details
    summary_details = {
        "Data Scientist": [
            "Experienced Data Scientist with a strong background in machine learning and statistical analysis. Skilled in using Python, R, and SQL to extract insights from complex datasets.",
            "Passionate about leveraging data to drive business decisions and improve operational efficiency. Proven track record of delivering high-impact data solutions."
        ],
        "Software Engineer": [
            "Highly skilled Software Engineer with experience in developing scalable and robust software applications. Proficient in multiple programming languages, including Java, Python, and JavaScript.",
            "Strong problem-solving abilities and a keen eye for detail, ensuring high-quality code and efficient solutions. Experienced in working with agile methodologies and collaborating with cross-functional teams."
        ],
        "Product Manager": [
            "Results-driven Product Manager with a proven track record of successfully launching products in competitive markets. Skilled in market analysis, product roadmapping, and stakeholder management.",
            "Strong leadership and communication skills, with experience in leading cross-functional teams. Passionate about understanding customer needs and translating them into successful products."
        ],
        "Financial Analyst": [
            "Detail-oriented Financial Analyst with experience in financial modeling, forecasting, and reporting. Skilled in using advanced Excel functions and financial software to analyze data and generate insights.",
            "Strong analytical and problem-solving skills, with a focus on driving business growth and efficiency. Experienced in collaborating with cross-functional teams to develop strategic financial plans."
        ],
        "Marketing Specialist": [
            "Creative Marketing Specialist with experience in developing and executing successful marketing campaigns. Skilled in digital marketing, social media management, and content creation.",
            "Strong analytical skills, with experience in using data to optimize marketing strategies and improve ROI. Experienced in collaborating with design and sales teams to create cohesive marketing plans."
        ]
    }

    # Initialize data dictionary
    data = {
        "full_name": [],
        "country": [],
        "country_full_name": [],
        "province": [],
        "city": [],
        "personal_emails": [],
        "personal_numbers": [],
        "URL": [],
        "gender": [],
        "headline": [],
        "summary": [],
        "industry": [],
        "experiences": [],
        "education": [],
        "skills": [],
        "certifications": [],
        "languages": [],
        "university": [],
        "company": []
    }

    # Generate data for each record
    for i in range(num_records):
        country = np.random.choice(list(countries_data.keys()))
        country_info = countries_data[country]

        full_name = f"{np.random.choice(country_info['first_names'])} {np.random.choice(country_info['last_names'])}"
        data["full_name"].append(full_name)
        data["country"].append(country)
        data["country_full_name"].append(country)
        data["province"].append(np.random.choice(country_info['provinces']))
        data["city"].append(np.random.choice(country_info['cities']))
        data["personal_emails"].append(f"{full_name.lower().replace(' ', '.')}@example.com")
        data["personal_numbers"].append(f"+{np.random.randint(1, 99)}{np.random.randint(100000000, 999999999)}")
        data["URL"].append(f"https://www.linkedin.com/in/{full_name.lower().replace(' ', '-')}")
        data["gender"].append(np.random.choice(["Male", "Female", "Other"]))

        # Assign job title and related details
        job_title = np.random.choice(list(job_titles.keys()))
        data["headline"].append(job_title)
        data["skills"].append(", ".join(job_titles[job_title]["skills"]))
        data["certifications"].append(", ".join(job_titles[job_title]["certifications"]))

        # Assign industry and other details
        data["industry"].append(np.random.choice(["Technology", "Finance", "Healthcare", "Education", "Retail"]))

        # Assign education details
        education = np.random.choice(education_details[country])
        data["education"].append(education)
        data["university"].append(education["institution"])  # Make university column match institution

        # Assign experience details with dynamic company names
        experience = np.random.choice(experience_details[job_title])
        experience["company"] = np.random.choice(country_info["companies"])
        data["experiences"].append(experience)

        data["summary"].append(np.random.choice(summary_details[job_title]))
        data["languages"].append(np.random.choice(country_info['languages']))

        # Assign company
        data["company"].append(np.random.choice(country_info['companies']))

    return pd.DataFrame(data)


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
    Executes the search and enrichment process using dummy data.
    """
    dummy_data = generate_dummy_data()

    # Apply filters
    if query:
        dummy_data = dummy_data[dummy_data['headline'].str.contains(query, case=False, na=False)]

    if country_filter:
        dummy_data = dummy_data[dummy_data['country'] == country_filter]

    if location_filter:
        dummy_data = dummy_data[dummy_data['city'] == location_filter]

    if province_filter:
        dummy_data = dummy_data[dummy_data['province'] == province_filter]

    if industry_filter:
        dummy_data = dummy_data[dummy_data['industry'] == industry_filter]

    if skills_filter:
        # Split the skills_filter into a list of skills
        skills_list = [skill.strip() for skill in skills_filter.split(',')]

        # Filter candidates who have all the specified skills
        dummy_data = dummy_data[dummy_data['skills'].apply(
            lambda x: all(skill.lower() in x.lower() for skill in skills_list)
        )]

    if languages_filter:
        dummy_data = dummy_data[dummy_data['languages'] == languages_filter]

    if university_filter:
        # Check both the university column and education field
        dummy_data = dummy_data[
            dummy_data['university'].str.contains(university_filter, case=False, na=False) |
            dummy_data['education'].str.contains(university_filter, case=False, na=False)
        ]

    if company_filter:
        # Check both the company column and experiences field
        dummy_data = dummy_data[
            dummy_data['experiences'].apply(
                lambda x: isinstance(x, dict) and company_filter.lower() in x.get('company', '').lower()
        )]

    if max_to_fetch:
        dummy_data = dummy_data.head(max_to_fetch)

    return dummy_data


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
            show_progress_bar=False
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
    st.title("Unified Candidate Search")
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
            search_query = st.text_input("Job Title/Position", placeholder="e.g. Data Scientist")

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
                company_filter = st.text_input("Company Name", placeholder="e.g. MTN")

            # Education and Industry filters
            edu_col1, edu_col2 = st.columns(2)
            with edu_col1:
                university_filter = st.text_input("University Name", placeholder="e.g. Stellenbosch")
            with edu_col2:
                industry_filter = st.text_input("Industry", placeholder="e.g. Accounting")

            # Skills and Certifications filters
            skills_col1, skills_col2 = st.columns(2)
            with skills_col1:
                skills_filter = st.text_input("Skills", placeholder="e.g. SQL, Python")
            with skills_col2:
                certifications_filter = st.text_input("Certifications", placeholder="e.g. Azure")

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
                    if results.empty:
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
                        experiences = row['experiences']
                        if isinstance(experiences, list):
                            for exp in experiences:
                                st.markdown(f"**Role:** {exp['role']} | **Company:** {exp['company']} | **From:** {exp['from']} | **To:** {exp['to']} | **Duration:** {exp['duration']}")
                        else:
                            st.markdown(f"**Role:** {experiences['role']} | **Company:** {experiences['company']} | **From:** {experiences['from']} | **To:** {experiences['to']} | **Duration:** {experiences['duration']}")

                    # In the main() function, where education details are displayed:
                    if pd.notnull(row['education']) and row['education']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        if isinstance(row['education'], dict):
                            edu = row['education']
                            st.markdown(f"**Institution:** {edu.get('institution', '')} | **Degree:** {edu.get('degree', '')} | **From:** {edu.get('from', '')} | **To:** {edu.get('to', '')}")
                        else:
                            # Fallback for old format if needed
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
                        experiences = row['experiences']
                        if isinstance(experiences, dict):
                            st.markdown(f"**Role:** {experiences.get('role', '')} | **Company:** {experiences.get('company', '')} | **From:** {experiences.get('from', '')} | **To:** {experiences.get('to', '')} | **Duration:** {experiences.get('duration', '')}")
                        elif isinstance(experiences, str):
                            for exp in experiences.split('\n'):
                                st.markdown(f"- {exp}")
                        elif isinstance(experiences, list):
                            for exp in experiences:
                                st.markdown(f"**Role:** {exp.get('role', '')} | **Company:** {exp.get('company', '')} | **From:** {exp.get('from', '')} | **To:** {exp.get('to', '')} | **Duration:** {exp.get('duration', '')}")

                    if pd.notnull(row['education']) and row['education']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        if isinstance(row['education'], dict):
                            edu = row['education']
                            st.markdown(f"**Institution:** {edu.get('institution', '')} | **Degree:** {edu.get('degree', '')} | **From:** {edu.get('from', '')} | **To:** {edu.get('to', '')}")
                        else:
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
