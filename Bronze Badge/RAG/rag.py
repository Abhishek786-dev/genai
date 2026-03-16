import logging
from src.document_load import DocumentLoader
from src.embedding import EmbeddingGenerator
from src.venctor_store import VectorStore
from logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)
logger.info("RAG Application started")

if __name__ == "__main__":
    # loader = DocumentLoader()
    # documents = loader.load()

    # embedding_generator = EmbeddingGenerator()
    # embeddings = embedding_generator.generate_embeddings(documents)

    # vector_store = VectorStore()
    # vector_store.add_embeddings(documents, embeddings)

    # logger.info("RAG Application finished")

    logger.info("Using Rag")
    rag = VectorStore()
    query = "what is difference between RISC and CISC process/controller ?"
    context = rag.search(query)
    print(context)
    result = rag.generate_llm_response(context, query)
    logger.info(f"Search Result: {result}")
    logger.info("RAG Application finished")
