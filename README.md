# Resume Screening Agent

This project is a local resume screening agent. It uses semantic search with sentence embeddings to find top matching resumes for a given job description, and then uses a local language model to evaluate and explain the match.

## Features

- **PDF and Text Parsing:** Extract text from PDF resumes or read plain text resumes.
- **Sentence Embeddings:** Embed texts using a Hugging Face sentence transformer (e.g., `all-MiniLM-L6-v2`).
- **FAISS Vector Store:** Index resume embeddings for efficient similarity search.
- **Semantic Search:** Retrieve the top-K resumes matching a job description.
- **Local LLM Scoring:** Use a local transformer model (e.g., FLAN-T5) to score and explain each resume-job match.
- **CPU-only Inference:** All models run locally on CPU for broad compatibility.

<!-- ## Folder Structure -->

