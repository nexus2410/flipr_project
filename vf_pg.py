import streamlit as st
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FIREBASE_KEY")

email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w{2,4}$'
password_pattern = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$'

def login_with_email_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return True, data
    else:
        return False, response.json()['error']['message']

def create_user_with_email_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return True, data
    else:
        return False, response.json()['error']['message']

st.title("User Authentication")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

col1, col2, col3, col4, col5 = st.columns(5)

def validate_email_password(email, password):
    if not re.match(email_pattern, email):
        return False, "Invalid email format."
    if not re.match(password_pattern, password):
        return False, "Password must be at least 6 characters, include letters and numbers."
    return True, ""

with col1:
    if st.button("Login"):
        valid, msg = validate_email_password(email, password)
        if not valid:
            st.error(f"❌ {msg}")
        else:
            success, result = login_with_email_password(email, password)
            if success:
                st.success("✅ Login successful!")
                st.session_state["user"] = result  # Store user info in session
                st.switch_page("pages/chatbot02.py")  # Redirect to chatbot page
            else:
                st.error(f"❌ Login failed: {result}")

with col5:
    if st.button("Create Account"):
        valid, msg = validate_email_password(email, password)
        if not valid:
            st.error(f"❌ {msg}")
        else:
            success, result = create_user_with_email_password(email, password)
            if success:
                st.success("✅ User created successfully!")
            else:
                st.error(f"❌ User creation failed: {result}")
