import streamlit as st
import pandas as pd
import os, re, base64, warnings, torch
from collections import Counter 
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from rapidfuzz import fuzz
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

warnings.filterwarnings("ignore")

device = torch.device("cpu")
model  = SentenceTransformer("all-mpnet-base-v2").to(device)
kw_model = KeyBERT()  # or KeyBERT(model) to reuse same embeddings


def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_email(text: str):
    m = re.search(r"\S+@\S+", text)
    return m.group(0) if m else ""


def extract_keywords_from_jd(jd_text: str, top_n: int = 50):
    jd_clean = re.sub(r"\s+", " ", jd_text.strip())
    kws = kw_model.extract_keywords(
        jd_clean,
        top_n=top_n,
        stop_words="english",
        use_mmr=True,
        diversity=0.7,
    )
    return [(kw.lower(), w) for kw, w in kws]

def keyword_coverage(resume_text: str, keywords):
    text = resume_text.lower()
    hit, total = 0.0, 0.0
    for kw, w in keywords:
        total += w
        if kw in text or fuzz.partial_ratio(kw, text) >= 85:
            hit += w
    return hit / total if total else 0.0


def compare_to_jd(jd_text: str, resume_text: str) -> float:
    jd_emb  = model.encode(jd_text,     convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    cos     = util.pytorch_cos_sim(jd_emb, res_emb).item()

    sem = max(0, min((cos - 0.25) / 0.5, 1))
    if sem > 0.80:
        sem = min(1, sem ** 1.25 + 0.05)

    kw_ratio = keyword_coverage(resume_text, extract_keywords_from_jd(jd_text))
    return round((0.60 * sem + 0.40 * kw_ratio) * 100, 2)


def process_resumes(uploaded_files, jd_text):
    rows = []
    for f in uploaded_files:
        tmp = f"/tmp/{f.name}"
        with open(tmp, "wb") as out: out.write(f.read())
        text = extract_text_from_pdf(tmp) if f.name.endswith(".pdf") else extract_text_from_docx(tmp)
        rows.append(
            {
                "File Name": f.name,
                "Email": extract_email(text),
                "Score (%)": compare_to_jd(jd_text, text),
            }
        )
    return pd.DataFrame(rows).sort_values("Score (%)", ascending=False)


st.set_page_config(
    page_title="Resume Ranker",
    page_icon="🔥",
    layout="centered",
)

# ---------- global CSS ----------
st.markdown(
    """
    <style>
        /* soft mint background */
        body {background: radial-gradient(circle at 50% 0%, #e8f4f1 0%, #f6fdfc 60%);}

        /* card-like main block */
        .main .block-container {background: #ffffff; padding: 2.5rem 4rem; border-radius: 12px;
                                 box-shadow: 0 4px 18px rgba(0,0,0,.05);}

        /* headings + fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html,body,div,span {font-family: 'Inter', sans-serif;}

        h1 {font-weight: 700; margin-bottom: .25rem;}
        h2 {font-weight: 600; margin-top: 1.6rem;}

        /* text-area keeps fixed height, no resize grip */
        textarea {resize: none !important; height: 120px !important;}

        /* nice primary button */
        .stButton>button {
            background:#116149; color:#fff; font-weight:600; border-radius:8px;
            padding:.6rem 2.2rem; border:none; transition: background .2s ease;
        }
        .stButton>button:hover {background:#0e523d;}

        /* drop-zone tweak */
        .stFileUploader>div>div {border:2px dashed #B6CBC3;}
        .stFileUploader span {color:#666; font-weight:500;}

        /* AgGrid – hide menu / resize handles */
        .ag-header-cell-menu-button, .ag-header-cell-resize {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- title bar ----------
col1, col2 = st.columns([1, 0.07])
with col1:
    st.markdown("<h1>Resume Ranker </h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='font-size:2.2rem; line-height:1.1;'>🔍</h1>", unsafe_allow_html=True)

# ---------- tabs ----------
tab_upload, tab_results = st.tabs(["📥 Upload & JD", "📊 Results"])

# --------------------------- Tab 1  (Upload) ------------------------------ #
with tab_upload:
    st.subheader("Step 1: Upload Resumes")
    resumes = st.file_uploader(
        "Upload multiple PDF/DOCX resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.subheader("Step 2: Paste Job Description")
    jd_text = st.text_area("e.g. We are hiring a backend Python developer with experience in …")

    run = st.button("🚀  Rank Resumes")
    if run and resumes and jd_text.strip():
        with st.spinner("Scoring…"):
            df = process_resumes(resumes, jd_text)
        st.session_state["rank_df"] = df
        st.success("✅ Ranking complete!")
        st.switch_page("/")          # instantly jump to “Results” tab below

# --------------------------- Tab 2  (Results) ----------------------------- #
with tab_results:
    if "rank_df" not in st.session_state:
        st.info("Upload resumes & a JD on the first tab, then click **Rank Resumes**.")
    else:
        st.subheader("Top Matching Resumes")
        df = st.session_state["rank_df"]

        # ---- AgGrid (prettier than st.dataframe) ----
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=False, filter=False, resizable=False,
                                    sortable=False, suppressMenu=True)
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, height=360, fit_columns_on_grid_load=True)

        # If you prefer vanilla Streamlit table:
        # st.dataframe(df, hide_index=True, use_container_width=True)
