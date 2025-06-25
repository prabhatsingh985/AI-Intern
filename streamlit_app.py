import streamlit as st
import os
from src.resume_parser import parse_resume
from src.retriever import ResumeRetriever
from src.scorer import ResumeScorer

st.title("🔍 Resume Screening Agent")

# Upload job description
job_description = st.text_area("Paste the Job Description:", height=200)

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resumes (.pdf/.txt)", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Screen Resumes"):

    if not job_description or not uploaded_files:
        st.warning("Please provide both job description and resumes.")
    else:
        resume_texts = []
        resume_names = []

        for uploaded_file in uploaded_files:
            try:
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                text = parse_resume(f"temp_{uploaded_file.name}")
                resume_texts.append(text)
                resume_names.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Failed to parse {uploaded_file.name}: {e}")
            finally:
                os.remove(f"temp_{uploaded_file.name}")

        retriever = ResumeRetriever()
        retriever.build_index(resume_texts)
        search_results = retriever.search(job_description, top_k=3)

        scorer = ResumeScorer()

        st.markdown(f"### 🏆 Top 3 Matching Resumes:")
        for rank, (idx, distance) in enumerate(search_results, start=1):
            resume_name = resume_names[idx]
            resume_text = resume_texts[idx]
            evaluation = scorer.evaluate(job_description, resume_text)
            st.subheader(f"Rank {rank}: {resume_name}")
            st.write(f"**Similarity Distance**: {distance:.4f}")
            st.markdown("**Evaluation:**")
            st.code(evaluation)
            st.markdown("---")
