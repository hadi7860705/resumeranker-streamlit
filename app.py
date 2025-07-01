import streamlit as st
import pandas as pd
import os, re, base64, warnings, torch
from collections import Counter  # only if you still need it elsewhere
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
st.set_page_config(page_title="Resume Ranker", page_icon="🔥", layout="wide")


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

    </style>
    """,
    unsafe_allow_html=True
)


with open("avialdo.jpeg", "rb") as f:
    logo_data = f.read()
    encoded = base64.b64encode(logo_data).decode()
    st.markdown(f'<p style="text-align:center;"><img src="data:image/jpeg;base64,{encoded}" width="400"/></p>', unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:#000000;'>Resume Ranker</h2>", unsafe_allow_html=True)
st.write("---")

st.subheader("📌 Step 1: Provide Job Description")
jd_col1, jd_col2 = st.columns(2)

jd_text = ""
with jd_col1:
    jd_text_area = st.text_area("Paste JD here", height=250)
with jd_col2:
    jd_file = st.file_uploader("Upload JD (.txt or .docx)", type=["txt", "docx"])

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

st.subheader("📂 Step 2: Upload Resume Files (.pdf or .docx)")
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)


if st.button("🚀 Rank Resumes"):
    if jd_text and uploaded_files:
        with st.spinner("Analyzing resumes..."):
            df = process_resumes(uploaded_files, jd_text)
        st.success("✅ Ranking complete!")

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            editable=False,
            filter=False,
            resizable=True,    # user can widen if necessary
            sortable=False,
            suppressMenu=True  # removes right-click “Format / Autosize…”
        )
        gb.configure_grid_options(
            suppressContextMenu=True,
            domLayout='normal'   # normal = we control height below
        )
        gridOptions = gb.build()

        n_rows   = max(len(df), 1)
        grid_h   = min(max(40 * n_rows + 60, 200), 600)

        # ---- render ----
        AgGrid(
            df,
            gridOptions=gridOptions,
            theme="material",
            height=grid_h,
            update_mode=GridUpdateMode.NO_UPDATE,
            enable_enterprise_modules=False
        )
    else:
        st.error("Please upload at least one resume **and** provide the JD.")
