import re
import string
from transformers import pipeline

class ResumeScorer:
    """
    Uses a local language model to evaluate how well a resume matches a job description,
    producing a natural-language explanation. Falls back to keyword-overlap logic if model
    output is empty/trivial.
    """
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        """
        Initialize the scorer with a Hugging Face transformer pipeline.

        Args:
            model_name (str): Name of the model for text2text generation.
                              E.g., 'google/flan-t5-small'. Make sure it's downloaded locally.
        """
        # Use a text-to-text generation pipeline. device=-1 ensures CPU-only.
        self.pipeline = pipeline(
            'text2text-generation',
            model=model_name,
            tokenizer=model_name,
            device=-1
        )
        # A minimal stopword list for fallback keyword extraction
        self.stopwords = {
            'and', 'or', 'the', 'with', 'in', 'to', 'for', 'of', 'on', 'by', 'a', 'an',
            'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
            'experience', 'experiences', 'skill', 'skills', 'using', 'use', 'working', 'work',
            'responsibilities', 'responsibility', 'including', 'etc'
        }

    def _extract_keywords(self, text: str) -> set:
        """
        Very simple keyword extractor: lowercases, removes punctuation, splits by whitespace,
        filters out stopwords and short tokens.

        Args:
            text (str): Input text.

        Returns:
            set: A set of keyword tokens.
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Split into tokens
        tokens = text.split()
        # Filter: remove stopwords and short tokens
        keywords = {
            tok for tok in tokens
            if len(tok) > 3 and tok not in self.stopwords
        }
        return keywords

    def evaluate(self, job_description: str, resume_text: str) -> str:
        """
        Evaluate the match between job description and resume text using the LLM,
        requesting a natural-language explanation. If the model's output is empty,
        trivial, or only numeric, use a fallback keyword-overlap explanation formatted
        as natural sentences.

        Args:
            job_description (str): The job description text.
            resume_text (str): The resume text to evaluate.

        Returns:
            str: The generated evaluation, either from LLM or fallback.
        """
        # Refined prompt: ask for a rating plus a concise natural-language explanation
        # starting with "The resume shows..." or "The resume lacks..."
        prompt = (
            "Job Description:\n"
            f"{job_description.strip()}\n\n"
            "Resume:\n"
            f"{resume_text.strip()}\n\n"
            "Question: On a scale of 0 to 10, how well does this resume match the job description? "
            "Provide a concise rating and a brief explanation in natural language. "
            "Begin the explanation with a phrase like “The resume shows...” if positive match, "
            "or “The resume lacks...” if weak match. "
            "Format example: “Rating: 8/10. The resume shows strong Python and ML experience, including ...”"
        )

        # Attempt LLM generation
        try:
            outputs = self.pipeline(prompt, max_new_tokens=256, do_sample=False)
            # Extract generated text
            result = outputs[0].get('generated_text', '').strip()
        except Exception:
            result = ""

        # Check if result is empty or trivial (e.g., only a number, or too short)
        # We'll consider trivial if it's empty, or matches purely numeric content, or very short (<20 chars)
        if not result or re.fullmatch(r'\d+(\.\d+)?', result) or len(result) < 20:
            # Fallback: simple keyword-overlap explanation
            jd_keywords = self._extract_keywords(job_description)
            res_keywords = self._extract_keywords(resume_text)
            common = jd_keywords.intersection(res_keywords)

            # Compute a naive score: ratio of overlapping keywords to total JD keywords, scaled to 0–10
            if jd_keywords:
                score_value = round((len(common) / len(jd_keywords)) * 10, 2)
            else:
                score_value = 0.0

            # Build a natural-language sentence
            if common:
                # Capitalize keywords for readability in explanation
                common_cap = [kw.capitalize() for kw in sorted(common)]
                # e.g., "Python, Machine, Learning"
                common_str = ", ".join(common_cap)
                explanation = (
                    f"The resume shows experience in {common_str}, which aligns with the job description."
                )
            else:
                explanation = (
                    "The resume does not show clear keyword overlap with the job description."
                )

            # Optionally comment on missing key skills: we could list top JD keywords not in resume
            missing = jd_keywords.difference(res_keywords)
            # For brevity, list up to 3 missing key terms
            if missing:
                missing_list = sorted(missing)
                missing_snippet = ", ".join(kw.capitalize() for kw in missing_list[:3])
                explanation += f" Missing key terms include: {missing_snippet}."
            # Construct fallback result string
            result = f"Rating: {score_value}/10. {explanation}"

        return result
