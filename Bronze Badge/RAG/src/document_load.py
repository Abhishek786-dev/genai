import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Any
from constant import FILE_PATH

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    DocumentLoader is responsible for loading documents from a specified file path.
    Attributes:
        file_path (str): Path to the directory containing the documents.
    """

    def __init__(self):
        self.file_path = FILE_PATH

    def load(self) -> List[Any]:
        """
        Load documents from the specified file path.
        Returns:
            List[Any]: List of loaded documents.
        """
        documents = []
        pdf_files = list(Path(self.file_path).glob("**/*.pdf"))
        logger.info(f"Starting Load documents from {self.file_path}")

        for file in pdf_files:
            try:
                logger.info(f"Loading PDF documents of {file.name}")
                loader = PyPDFLoader(str(file))
                loader = loader.load()
                documents.extend(loader)
            except Exception as e:
                logger.error(f"Error While loading documents: {e}")

        logger.info(
            f"Finished loading documents. Total documents loaded: {len(pdf_files)}"
        )

        return documents
