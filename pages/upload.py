import streamlit as st
import os
from datetime import datetime

DATA_DIR = r"F:\flipr\data"
RECORD_FILE = os.path.join(DATA_DIR, "records.txt")

os.makedirs(DATA_DIR, exist_ok=True)

def save_text_data(text, user_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_userid = user_id.replace(" ", "_").replace(os.sep, "_")
    user_dir = os.path.join(DATA_DIR, safe_userid)
    os.makedirs(user_dir, exist_ok=True)
    filename = f"{timestamp}_{safe_userid}.txt"
    filepath = os.path.join(user_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    with open(RECORD_FILE, "a", encoding="utf-8") as rec:
        rec.write(f"{timestamp},{safe_userid},{filename}\n")

def load_records(user_id=None):
    if not os.path.exists(RECORD_FILE):
        return []
    with open(RECORD_FILE, "r", encoding="utf-8") as rec:
        lines = rec.readlines()
    records = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 3:
            if user_id is None or parts[1] == user_id:
                records.append({"Timestamp": parts[0], "Filename": parts[2]})
    return records

st.title("Text Data Uploader")

# Only allow logged in users
if "user" not in st.session_state or not st.session_state["user"]:
    st.warning("You must be logged in to upload files. Please log in from the main page.")
    st.stop()

user_id = st.session_state["user"].get("localId") or st.session_state["user"].get("email")
user_text = st.text_area("Enter your text data:")

if st.button("Submit"):
    if not user_text.strip():
        st.warning("Please enter some text before submitting.")
    else:
        save_text_data(user_text, user_id)
        st.success("Text data saved successfully!")

# Add logout button
logout_col, _ = st.columns([1, 5])
with logout_col:
    if st.button("Log Out"):
        st.session_state.pop("user", None)
        st.success("Logged out successfully!")
        st.stop()

st.subheader("Upload Records")
records = load_records(user_id)
if records:
    st.table(records[::-1])
else:
    st.write("No records yet.")