#!/usr/bin/env python3

import os
from src.resume_parser import parse_resume
from src.retriever import ResumeRetriever
from src.scorer import ResumeScorer

def main():
    """
    Main function to run the resume screening agent.
    """
    # Load job description from file
    job_desc_path = os.path.join('data', 'job_description.txt')
    with open(job_desc_path, 'r', encoding='utf-8') as f:
        job_description = f.read()

    # Load and parse all resumes from the data/resumes directory
    resume_dir = os.path.join('data', 'resumes')
    resume_files = [f for f in os.listdir(resume_dir) if f.lower().endswith(('.txt', '.pdf'))]
    resume_texts = []
    resume_names = []
    for file in resume_files:
        file_path = os.path.join(resume_dir, file)
        try:
            text = parse_resume(file_path)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            continue
        resume_texts.append(text)
        resume_names.append(file)

    # Initialize the retriever and build the FAISS index with resume embeddings
    retriever = ResumeRetriever()
    retriever.build_index(resume_texts)

    # Perform semantic search to find top-3 matching resumes
    top_k = 3
    search_results = retriever.search(job_description, top_k=top_k)

    # Initialize the scorer (local LLM) for evaluation
    scorer = ResumeScorer()

    # Print and evaluate each retrieved resume
    print(f"Top {top_k} resumes matching the job description:")
    for rank, (idx, distance) in enumerate(search_results, start=1):
        resume_name = resume_names[idx]
        resume_text = resume_texts[idx]
        evaluation = scorer.evaluate(job_description, resume_text)
        print(f"\nRank {rank}: {resume_name} (Similarity distance: {distance:.4f})")
        print(evaluation)
        print("-" * 80)

if __name__ == '__main__':
    main()
