import streamlit as st
import pandas as pd
import os
import re
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import base64
import warnings
import re

import torch
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

warnings.filterwarnings("ignore")

device = torch.device("cpu")  # Fix for Streamlit + PyTorch meta tensor issue
model = SentenceTransformer('all-mpnet-base-v2')
model = model.to(device)

# ------------------- Utility Functions -------------------
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_email(text):
    match = re.search(r'\S+@\S+', text)
    return match.group(0) if match else ""

def extract_required_skills(jd_text):
    jd_text = jd_text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', jd_text)
    common_words = set(['the', 'and', 'for', 'with', 'you', 'are', 'our', 'this', 'your', 'will',
                        'have', 'job', 'who', 'that', 'from', 'can', 'not', 'all', 'able', 'work',
                        'we', 'as', 'to', 'in', 'on', 'of', 'is', 'be', 'a', 'an', 'or', 'it', 'they'])
    filtered = [word for word in words if word not in common_words]
    most_common = Counter(filtered).most_common(10)
    return [word for word, _ in most_common]

def keyword_match_score(resume_text, required_skills):
    resume_text = resume_text.lower()
    match_count = sum(1 for skill in required_skills if skill in resume_text)
    return match_count / len(required_skills) * 100 if required_skills else 0

from keybert import KeyBERT
kw_model = KeyBERT()

def extract_keywords_from_jd(jd_text: str, top_n: int = 30):
    """
    Return a list of (keyword, weight) pairs using MMR for diversity.
    """
    jd_text_clean = re.sub(r'\s+', ' ', jd_text.strip())
    keywords = kw_model.extract_keywords(
        jd_text_clean,
        top_n=top_n,
        stop_words="english",
        use_mmr=True,
        diversity=0.7
    )
    # lower-case keywords for case-insensitive matching
    return [(kw.lower(), weight) for kw, weight in keywords]


def compare_to_jd(jd_text, resume_text):
    keywords = extract_keywords_from_jd(jd_text, top_n=30)
    res_low = resume_text.lower()

    # --- Semantic similarity ---
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    raw_cos = util.pytorch_cos_sim(jd_emb, res_emb).item()
    sem = max(0, min((raw_cos - 0.25) / 0.5, 1))  # normalize to [0,1]

    # --- Boost excellent semantic alignment ---
    if sem > 0.8:
        sem = sem ** 1.45 + 0.15  # slightly more boost
    semantic_score = 100 * (sem ** 1.35)  # gentler rise for 0.6–0.8

    # --- Keyword match ---
    hit_w = sum(w for kw, w in keywords if kw in res_low)
    total_w = sum(w for _, w in keywords)
    kw = hit_w / total_w if total_w > 0 else 0

    # --- Keyword penalty ---
    keyword_penalty = 100 * ((1 - kw) ** 2.2)  # slight reduction in penalty

    # --- Final score ---
    final_score = semantic_score - 0.75 * keyword_penalty
    return round(max(0, min(final_score, 100)), 2)
    
def process_resumes(uploaded_files, jd_text):
    results = []
    for file in uploaded_files:
        filename = file.name
        file_path = f"/tmp/{filename}"
        with open(file_path, "wb") as f:
            f.write(file.read())
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            continue
        score = compare_to_jd(jd_text, text)
        email = extract_email(text)
        results.append({"File Name": filename, "Email": email, "Score (%)": score})
    df = pd.DataFrame(results)
    df = df.sort_values(by="Score (%)", ascending=False)
    return df

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Resume Ranker", page_icon="🔥", layout="wide")

# Display logo
with open("avialdo.jpeg", "rb") as f:
    logo_data = f.read()
    encoded = base64.b64encode(logo_data).decode()
    st.markdown(f'<p style="text-align:center;"><img src="data:image/jpeg;base64,{encoded}" width="400"/></p>', unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:#E74C3C;'>Resume Ranker: Match Resumes to Job Description with AI</h2>", unsafe_allow_html=True)
st.write("---")

# Step 1: JD Input
st.subheader("📌 Step 1: Provide Job Description")
jd_col1, jd_col2 = st.columns(2)

jd_text = ""
with jd_col1:
    jd_text_area = st.text_area("Paste JD here (optional if uploading)")
with jd_col2:
    jd_file = st.file_uploader("Or upload JD (.txt or .docx)", type=["txt", "docx"])

if jd_file:
    file_path = f"/tmp/{jd_file.name}"
    with open(file_path, "wb") as f:
        f.write(jd_file.read())
    if jd_file.name.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
    elif jd_file.name.endswith(".docx"):
        jd_text = extract_text_from_docx(file_path)
elif jd_text_area.strip():
    jd_text = jd_text_area.strip()

# Step 2: Resume Upload
st.subheader("📂 Step 2: Upload Resume Files (.pdf or .docx)")
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

# Button
if st.button("🚀 Rank Resumes"):
    if jd_text and uploaded_files:
        with st.spinner("Analyzing resumes..."):
            df = process_resumes(uploaded_files, jd_text)
            st.success("✅ Ranking complete!")
            st.dataframe(df, use_container_width=True)
    else:
        st.error("Please upload at least one resume and provide the job description.")


