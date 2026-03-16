import logging
import numpy as np
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from constant import MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    EmbeddingGenerator is responsible for generating embeddings for documents.
    Attributes:
        model_name (str): Name of the pre-trained model to use for generating embeddings.
    """

    def __init__(self):
        self.model_name = MODEL_NAME
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.load_model()

    def load_model(self):
        """
        Load the pre-trained model for generating embeddings.
        """
        try:
            logger.info(f"Loading Embedding Model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding Model loaded successfully")
        except Exception as e:
            logger.error(f"Error while loading Embedding Model: {e}")

    def chunk_documents(self, documents: List[Any]) -> List[str]:
        """
        Chunk the given documents into smaller pieces.
        Args:
            documents (List[Any]): List of documents to chunk.
        Returns:
            List[str]: List of chunked documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Documents chunked into {len(chunks)} chunks")
        return [chunk.page_content for chunk in chunks]

    def generate_embeddings(self, documents: List[Any]) -> np.ndarray:
        """
        Generate embeddings for the given documents.
        Args:
            documents (List[Any]): List of documents to generate embeddings for.
        Returns:
            np.ndarray: Array of generated embeddings.
        """
        chunks = self.chunk_documents(documents)
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            logger.info(
                f"Embeddings generated successfully, {embeddings.shape[0]} embeddings created"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error while generating embeddings: {e}")
            return None
