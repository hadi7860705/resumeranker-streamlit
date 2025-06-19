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
warnings.filterwarnings("ignore")

model = SentenceTransformer('all-mpnet-base-v2')  

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

def compare_to_jd(jd_text, resume_text):
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()

    # Apply custom transformation to improve differentiation
    if similarity >= 0.7:
        score = round(similarity * 100, 2)
    elif similarity >= 0.5:
        score = round((similarity - 0.2) * 100, 2)
    elif similarity >= 0.3:
        score = round((similarity - 0.1) * 90, 2)
    else:
        score = round(similarity * 70, 2)

    return score
    
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


