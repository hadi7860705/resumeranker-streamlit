import streamlit as st
import os
import re
import pandas as pd
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


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
    common_words = set([
        'the', 'and', 'for', 'with', 'you', 'are', 'our', 'this', 'your', 'will',
        'have', 'job', 'who', 'that', 'from', 'can', 'not', 'all', 'able', 'work',
        'we', 'as', 'to', 'in', 'on', 'of', 'is', 'be', 'a', 'an', 'or', 'it', 'they'
    ])
    filtered = [word for word in words if word not in common_words]
    most_common = Counter(filtered).most_common(10)
    return [word for word, _ in most_common]


def keyword_match_score(resume_text, required_skills):
    resume_text = resume_text.lower()
    match_count = sum(1 for skill in required_skills if skill in resume_text)
    return match_count / len(required_skills) * 100 if required_skills else 0


def compare_to_jd(jd_text, resume_text):
    required_skills = extract_required_skills(jd_text)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
    keyword_score = keyword_match_score(resume_text, required_skills)
    return round((similarity * 100 * 0.7) + (keyword_score * 0.3), 2)


def process_resumes(uploaded_files, jd_text):
    results = []
    for file in uploaded_files:
        filename = file.name
        if filename.endswith(".pdf"):
            with open(f"/tmp/{filename}", "wb") as f:
                f.write(file.read())
            text = extract_text_from_pdf(f"/tmp/{filename}")
        elif filename.endswith(".docx"):
            with open(f"/tmp/{filename}", "wb") as f:
                f.write(file.read())
            text = extract_text_from_docx(f"/tmp/{filename}")
        else:
            continue

        score = compare_to_jd(jd_text, text)
        email = extract_email(text)

        results.append({
            "File Name": filename,
            "Email": email,
            "Score (%)": score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score (%)", ascending=False)
    return df


# Streamlit UI
st.title("📄 Resume Ranker")
st.write("Upload resumes and paste the job description to get ranked results.")

jd_text = st.text_area("📌 Paste Job Description Here")

uploaded_files = st.file_uploader("📂 Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("🚀 Rank Resumes"):
    if jd_text and uploaded_files:
        with st.spinner("Processing..."):
            df = process_resumes(uploaded_files, jd_text)
            st.success("Ranking complete!")
            st.dataframe(df)

            # Download option
            st.download_button(
                label="📥 Download Ranked Results as Excel",
                data=df.to_excel(index=False, engine='openpyxl'),
                file_name="ranked_candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("Please upload resumes and enter job description first.")




