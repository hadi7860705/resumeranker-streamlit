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
    
from rapidfuzz import fuzz
from keybert import KeyBERT
kw_model = KeyBERT()
TOP_N_KW = 50  

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


# ---------------- Keyword helpers -----------------
from rapidfuzz import fuzz
from keybert import KeyBERT

kw_model = KeyBERT(model)          # reuse existing SentenceTransformer
TOP_N_KW = 50                      # pull more keywords

def extract_keywords_from_jd(jd_text: str, top_n: int = TOP_N_KW):
    """
    Returns list of tuples (keyword, weight) sorted by KeyBERT relevance.
    """
    return kw_model.extract_keywords(jd_text, top_n=top_n, stop_words="english")

def keyword_coverage(resume_text: str, keywords):
    """
    Return ratio [0‒1] of JD keywords found in resume_text.
    Uses exact OR fuzzy partial match (RapidFuzz ≥85).
    Weights each keyword by KeyBERT weight.
    """
    text = resume_text.lower()
    matched_weight = 0.0
    total_weight   = 0.0
    for kw, w in keywords:
        total_weight += w
        if kw in text or fuzz.partial_ratio(kw, text) >= 85:
            matched_weight += w
    return matched_weight / total_weight if total_weight else 0.0
    
# --------------- Scoring function -----------------
def compare_to_jd(jd_text: str, resume_text: str) -> float:
    # ----- semantic similarity -----
    jd_emb  = model.encode(jd_text,     convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    raw_cos = util.pytorch_cos_sim(jd_emb, res_emb).item()

    # Normalise to [0,1] (shift & scale typical SBERT cosine range 0.25–0.75)
    sem = max(0.0, min((raw_cos - 0.25) / 0.5, 1.0))

    # Gentle boost for VERY high semantic (>0.80) so the best still pop
    if sem > 0.80:
        sem = min(1.0, sem ** 1.25 + 0.05)

    # ----- keyword coverage -----
    jd_keywords = extract_keywords_from_jd(jd_text)
    kw_ratio    = keyword_coverage(resume_text, jd_keywords)          # 0‒1

    # ----- final blend (60 % semantic, 40 % keyword) -----
    final = (0.60 * sem + 0.40 * kw_ratio) * 100                     # -> 0‒100
    return round(final, 2)
    
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
from st_aggrid import AgGrid, GridOptionsBuilder

st.markdown(
    """
    <style>
        /* 1️⃣ General rule */
        .stTextArea textarea {
            resize: none !important;          /* turn off user-resize */
            height: 75px !important;         /* keep constant size   */
            min-height: 75px !important;
            max-height: 75px !important;
            overflow-y: auto !important;      /* still scroll inside  */
        }

        /* 2️⃣ Chrome / Edge / Safari: hide the bottom-right grip */
        .stTextArea textarea::-webkit-resizer {
            display: none !important;
        }

        /* 3️⃣ Firefox has a different shadow part */
        .stTextArea textarea::-moz-resizer {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
    jd_text_area = st.text_area("Paste JD here (optional if uploading)", height=250)
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
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(
                editable=False,
                filter=False,
                resizable=False,
                sortable=False,
                suppressMenu=True
            )
            gb.configure_grid_options(suppressContextMenu=True)
            gridOptions = gb.build()

            AgGrid(df, gridOptions=gridOptions, enable_enterprise_modules=False)    
    else:
        st.error("Please upload at least one resume and provide the job description.")


