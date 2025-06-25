import fitz

def parse_resume(file_path: str) -> str:
    """
    Parse a resume file (PDF or text) and extract plain text.

    Args:
        file_path (str): Path to the resume file (PDF or .txt).

    Returns:
        str: Extracted text content of the resume.
    """
    text = ""
    # Check file extension to decide parsing method
    if file_path.lower().endswith('.pdf'):
        # Use PyMuPDF to extract text from PDF
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    elif file_path.lower().endswith('.txt'):
        # Read text file directly
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format for file: {file_path}")
    return text
