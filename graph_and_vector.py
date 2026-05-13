# Importing Libraries
import streamlit as st
from llm import embeddings
from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma


# -----------------------------
# Graph Store Setup
# -----------------------------
@st.cache_resource
def get_graph_store(uri, username, password, database):
    return Neo4jGraph(
        url=uri,
        username=username,
        password=password,
        database=database
    )

def get_graph_main():
    return get_graph_store(
        st.session_state["MAIN_NEO4J_URI"],
        st.session_state["MAIN_NEO4J_USERNAME"],
        st.session_state["MAIN_NEO4J_PASSWORD"],
        st.session_state["MAIN_NEO4J_DATABASE"]
    )

# -----------------------------
# Vector Store Setup
# -----------------------------
@st.cache_resource
def get_vector_store(collection_name, api_key, tenant, db):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        chroma_cloud_api_key=api_key,
        tenant=tenant,
        database=db
    )

def get_vector_main():
    return get_vector_store(
        st.session_state["COLLECTION_NAME"],
        st.session_state["CHROMA_API_KEY"],
        st.session_state["CHROMA_TENANT"],
        st.session_state["CHROMA_DB"]
    )
