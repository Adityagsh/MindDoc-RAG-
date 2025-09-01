import warnings
import logging
import streamlit as st
import os
# Local modules
from chat import display_chat_history, handle_user_input
from file_handler import upload_files, save_uploaded_files, extract_text_from_file
from vectorstore import load_vectorstore
from llm import get_llm_chain
from chroma_inspector import inspect_chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Silence noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title=" MindDoc-RAG",     
)



# App title
st.title("Chat with your Document! (🧠+📄) ")

# Step 1: Upload File + wait for submit
uploaded_files, submitted = upload_files()

# Step 2: If user clicks submit, update vectorstore
if submitted and uploaded_files:
    with st.spinner(" Updating vector database..."):
        vectorstore = load_vectorstore(uploaded_files)
        st.session_state.vectorstore = vectorstore


# Step 3: Display vectorstore inspector (Sidebar)
if "vectorstore" in st.session_state:
    inspect_chroma(st.session_state.vectorstore)

# Step 4: Display old chat messages
display_chat_history()

# Step 5: Handle new prompt input
if "vectorstore" in st.session_state:
    handle_user_input(get_llm_chain(st.session_state.vectorstore))

# Step 6: Chat history export




