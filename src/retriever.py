from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ResumeRetriever:
    """
    Handles embedding of resume texts and semantic search using FAISS.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the retriever with a sentence transformer model.

        Args:
            model_name (str): Name of the Hugging Face sentence transformer model.
        """
        # Load sentence transformer model (for example, all-MiniLM-L6-v2)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = None

    def build_index(self, texts: list):
        """
        Build a FAISS index from a list of texts.

        Args:
            texts (list): List of strings, each being a resume text.
        """
        # Compute embeddings for all resume texts
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        # Convert to numpy array and ensure float32 type for FAISS
        embeddings = np.array(embeddings).astype('float32')
        # Initialize FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        # Add embeddings to index
        self.index.add(embeddings)
        # Store texts for reference (ordered corresponding to embeddings)
        self.texts = texts

    def search(self, query: str, top_k: int = 3):
        """
        Search for top_k resumes that best match the query.

        Args:
            query (str): Query string (job description).
            top_k (int): Number of top matches to return.

        Returns:
            list of tuples: Each tuple is (index, distance).
                             'index' corresponds to position in the texts list.
        """
        if self.index is None:
            raise ValueError("The FAISS index has not been built. Call build_index() first.")
        # Encode the query into an embedding
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        # Prepare results: pair (index, distance) for each top match
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((int(idx), float(dist)))
        return results
