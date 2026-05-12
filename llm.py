# Importing Libraries
import os
import logging as py_logging
from transformers.utils import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
logging.set_verbosity_error()
logging.disable_progress_bar()
py_logging.getLogger("transformers").setLevel(py_logging.ERROR)

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()


# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)


# -----------------------------
# Embedding Model Setup
# -----------------------------
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

local_model = get_embedding_model()

class LangChainNomicWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.encode(prefixed).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(f"search_query: {text}").tolist()

embeddings = LangChainNomicWrapper(local_model)
