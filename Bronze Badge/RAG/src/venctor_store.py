import os
import faiss
import pickle
import logging
from typing import List, Any
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from src.embedding import EmbeddingGenerator
from langchain_community.llms import Ollama
from constant import (
    FAISS_STORE,
    MODEL_NAME,
    TOP_K,
    GROQ_API_KEY,
    GROQ_MODEL,
    THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logger = logging.getLogger(__name__)


class VectorStore:
    """
    VectorStore is responsible for storing and retrieving embeddings.
    Attributes:
        model_name (str): Name of the pre-trained model to use for generating embeddings.
    """

    def __init__(self):
        self.persist_dir = FAISS_STORE
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.metadata = []
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=1000,
        )
        self.load_index()

    def load_index(self):
        """
        Load the FAISS index and metadata from the specified directory.
        """
        try:
            if os.path.exists(self.persist_dir):
                logger.info(f"Loading FAISS index from {self.persist_dir}")
                self.index = faiss.read_index(
                    os.path.join(self.persist_dir, "faiss_index")
                )
                with open(os.path.join(self.persist_dir, "metadata.pkl"), "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info("FAISS index loaded successfully")
            else:
                logger.info(
                    f"No existing FAISS index found at {self.persist_dir}. A new index will be created."
                )
                self.index = None
                self.metadata = []
        except Exception as e:
            logger.error(f"Error while loading FAISS index: {e}")
            self.index = None
            self.metadata = []

    def save_index(self):
        """Save the FAISS index and metadata to the specified directory."""
        try:
            if self.index is not None:
                logger.info(f"Saving FAISS index to {self.persist_dir}")
                os.makedirs(self.persist_dir, exist_ok=True)
                faiss.write_index(
                    self.index, os.path.join(self.persist_dir, "faiss_index")
                )
                with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
                    pickle.dump(self.metadata, f)
                logger.info("FAISS index saved successfully")
            else:
                logger.warning("No FAISS index to save.")
        except Exception as e:
            logger.error(f"Error while saving FAISS index: {e}")

    def add_embeddings(self, documents: List[Any], embeddings: List[Any]):
        """
        Add embeddings for the given documents to the FAISS index.
        Args:
            documents (List[Any]): List of documents to add embeddings for.
            embeddings (List[Any]): List of embeddings to add.
        """
        try:
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            self.metadata.extend([doc.page_content for doc in documents])
            logger.info(f"Added {len(documents)} documents to the FAISS index")
            self.save_index()
        except Exception as e:
            logger.error(f"Error while adding embeddings to FAISS index: {e}")

    def search(self, query: str) -> List[str]:
        """
        Search for relevant documents based on the given query.
        Args:
            query (str): Query string to search for.
        Returns:
            List[str]: List of relevant documents based on the query.
        """
        try:
            query_embedding = self.embedding_model.encode([query]).astype("float32")
            logger.info(f"Searching for relevant documents for the query: {query}")
            distances, indices = self.index.search(query_embedding, TOP_K)
            results = []
            for distance, index in zip(distances[0], indices[0]):
                # Only include results that are within the specified distance threshold
                if index < len(self.metadata):
                    results.append(
                        {
                            "index": index,
                            "distance": distance,
                            "metadata": self.metadata[index],
                        }
                    )
            logger.info(
                f"Search completed. Found {len(results)} relevant documents for the query."
            )
            return (
                "\n\n".join([doc["metadata"] for doc in results])
                if results
                else "No relevant documents found."
            )
        except Exception as e:
            logger.error(f"Error while searching for relevant documents: {e}")
            return []

    def generate_llm_response(self, context: str, query: str) -> str:
        """
        Generate an LLM response based on the given context.
        Args:
            context (str): Context string to generate a response for.
        Returns:
            str: Generated response from the LLM.
        """
        try:
            logger.info("Generating LLM response based on the retrieved context.")
            prompt = f"""Use the following context to answer the question concisely.
                        context:{context}
                        query:{query}
                        Answer:"""
            logger.info("Invoking LLM to generate response.")
            response = self.llm.invoke(prompt)
            logger.info("LLM response generated successfully.")
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error while generating LLM response: {e}")
            return "Error generating response."
