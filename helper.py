# Importing Libraries
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompts import KEYWORD_PROMPT_TEMPLATE
from graph_and_vector import get_vector_main, get_graph_main


# -----------------------------
# Global Constants
# -----------------------------
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50
BEST_CHUNK_SIZE = 3
COURSE_INFERENCE_K = 50
TOP_K_SCHOOLS = 1
VECTOR_RETRIVAL_WEIGHT = 0.3
GRAPH_RETRIVAL_WEIGHT = 0.7


# -----------------------------
# Query Augmentation - Chunking + Rewriting
# -----------------------------
def chunk_resume(resume: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    if not resume or len(resume.strip()) < 50:
        return []
    return splitter.split_text(resume)


def extract_resume_keywords(llm_model) -> list[str]:
    keyword_prompt = ChatPromptTemplate.from_template(KEYWORD_PROMPT_TEMPLATE)
    keyword_chain = keyword_prompt | llm_model

    resume_chunks = st.session_state.get("resume_chunks", [])
    if not resume_chunks:
        return []

    best_chunk = "\n".join(resume_chunks[:BEST_CHUNK_SIZE])
    try:
        result = keyword_chain.invoke({"resume": best_chunk}).content
        keywords = [k.strip() for k in result.split(",") if k.strip()]
        return keywords[:12]
    except Exception:
        return []


# -----------------------------
# School Inference
# -----------------------------
def infer_school_context(keywords):
    try:
        vector_db = get_vector_main()
        graph_db = get_graph_main()

        resume_docs = vector_db.similarity_search(st.session_state.resume_data, k=COURSE_INFERENCE_K, filter={"type": "Course"})
        resume_schools = [d.metadata.get("school") for d in resume_docs if d.metadata.get("school")]
        fold1_counts = {}
        for school in resume_schools:
            fold1_counts[school] = fold1_counts.get(school, 0) + 1

        fold2_counts = {}
        if keywords:
            cypher_query = """
            UNWIND $keywords AS skill_name
            MATCH (sk:Skill)<-[:TEACHES]-(c:Course)<-[:HAS_COURSE]-(s:School)
            WHERE toLower(sk.name) CONTAINS toLower(skill_name)
            RETURN s.name AS school, count(c) AS course_count
            ORDER BY course_count DESC
            """
            try:
                graph_results = graph_db.query(cypher_query, params={"keywords": keywords})
                fold2_counts = {r['school']: r['course_count'] for r in graph_results}
            except Exception as e1:
                fold2_counts = {}

        s1 = pd.Series(fold1_counts) * VECTOR_RETRIVAL_WEIGHT
        s2 = pd.Series(fold2_counts) * GRAPH_RETRIVAL_WEIGHT
        combined_scores = s1.add(s2, fill_value=0).sort_values(ascending=False)
        top_k_schools = combined_scores.head(TOP_K_SCHOOLS).index.tolist()
        if len(top_k_schools) != 0:
            return " ; ".join(top_k_schools)
        else:
            return 'No specific school details inferred.'
    except Exception as e:
        return 'No specific school details inferred.'
