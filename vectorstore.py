from langchain_community.vectorstores import Chroma  # Vector database
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader  # Loaders
from langchain.embeddings import HuggingFaceEmbeddings  # Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splitter
from file_handler import save_uploaded_files  # Custom file saving
import os
import sys

# --- SQLite fallback handling (fix for Chroma + old sqlite3) ---
try:
    import sqlite3
    from sqlite3 import sqlite_version_info

    if sqlite_version_info < (3, 35, 0):
        # Replace sqlite3 with pysqlite3 if installed
        import pysqlite3
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # If sqlite3 is missing or broken, fallback to pysqlite3
    try:
        import pysqlite3
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        raise ImportError("Neither sqlite3 >= 3.35.0 nor pysqlite3 is available. Please install pysqlite3-binary.")

# Directory where the Chroma vector DB is stored
PERSIST_DIR = "./chroma_store"


def load_vectorstore(uploaded_files):
    """Load or create a Chroma vectorstore from uploaded files (PDF/DOCX)."""
    paths = save_uploaded_files(uploaded_files)

    docs = []
    for path in paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue  # Skip unsupported files
        docs.extend(loader.load())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if vectorstore already exists
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    return vectorstore
