# Importing Libraries
import os
from llm import embeddings
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma

load_dotenv()


# -----------------------------
# Graph Store Setup
# -----------------------------
graph_main = Neo4jGraph(
    url = os.getenv("MAIN_NEO4J_URI"),
    username = os.getenv("MAIN_NEO4J_USERNAME"),
    password = os.getenv("MAIN_NEO4J_PASSWORD"),
    database = os.getenv("MAIN_NEO4J_DATABASE")
)

# -----------------------------
# Vector Store Setup
# -----------------------------
vector_store = Chroma(
    collection_name="entities",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DB"),
)
