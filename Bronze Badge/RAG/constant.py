import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DIR_PATH, "files")


MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
THRESHOLD = 0.8

FAISS_STORE = os.path.join(DIR_PATH, "faiss_store")


GROQ_API_KEY = "USE YOUR API KEY"
GROQ_MODEL = "llama-3.1-8b-instant"
