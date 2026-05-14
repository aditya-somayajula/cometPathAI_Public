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
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Model Options
# -----------------------------
MODEL_OPTIONS = [
    "Groq - Llama 3.3 70B Versatile",
    "OpenAI - GPT-4o Mini",
    "Anthropic - Claude Sonnet 4.5",
    "Anthropic - Claude Haiku 4.5"
]
DEFAULT_MODEL = "Groq - Llama 3.3 70B Versatile"

# -----------------------------
# LLM Setup
# -----------------------------
@st.cache_resource
def get_llm(model_name, model_api):
    if model_name == "Groq - Llama 3.3 70B Versatile":
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5, groq_api_key=model_api)
    elif model_name == "OpenAI - GPT-4o Mini":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=model_api)
    elif model_name == "Anthropic - Claude Sonnet 4.5":
        llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.5, anthropic_api_key=model_api)
    elif model_name == "Anthropic - Claude Haiku 4.5":
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.5, anthropic_api_key=model_api)
    else:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5, groq_api_key=model_api)
    return llm


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


# -----------------------------
# Reranker Setup
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
