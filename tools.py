# Importing Libraries
import json
import numpy as np
import streamlit as st
from llm import get_llm, reranker
from langchain.tools import tool
from rank_bm25 import BM25Okapi
from prompts import CYPHER_GENERATION_TEMPLATE
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from graph_and_vector import get_graph_main, get_vector_main


# -----------------------------
# Vector Store Tool
# -----------------------------
@tool
def vector_search(query: str):
    """
    Use this to find Schools, Programs, or Courses based on semantic similarity to
    skills or interests or resume or goal.
    """
    docs = get_vector_main().similarity_search(query, k=10)
    # return "\n -- \n".join([f"Content: {d.page_content}\nMetadata: {d.metadata}" for d in docs])
    return json.dumps([
    {
        "content": d.page_content,
        "metadata": d.metadata
    }
    for d in docs
])


# -----------------------------
# Graph Store Tool
# -----------------------------
@tool
def graph_search(question: str):
    """
        Use this tool to answer questions about the Neo4j knowledge graph containing
        schools, programs, certificates, courses, professors, skills, and their relationships.

        The tool accepts natural language questions and automatically generates,
        validates, and executes Cypher queries using the graph schema.

        Capabilities:
        - Find relationships between entities
        - Retrieve course, program, school, and professor information
        - Explore prerequisite and skill relationships
        - Discover programs associated with courses or skills
        - Identify professors, departments, and research areas
        - Recommend related courses or programs based on graph connections

        The tool returns structured graph-based results generated from Neo4j.

        Example questions:
        - "Which courses are related to machine learning?"
        - "Find professors researching artificial intelligence"
        - "What programs include data science courses?"
        - "Which skills are associated with CS 6360?"
        - "Show courses taught by professors in the computer science department"

        Do not provide raw Cypher queries as input.
        Input should always be a natural language question.
    """
    graph = get_graph_main()
    llm_model = get_llm(st.session_state.model_name, st.session_state.model_api_key)

    cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

    cypher_qa = GraphCypherQAChain.from_llm(
        llm_model,
        graph=graph,
        verbose=False,
        cypher_prompt=cypher_prompt,
        top_k=10,
        validate_cypher=True,
        allow_dangerous_requests=True,
        return_direct=False
    )

    response = cypher_qa.invoke({
        "query": question
    })
    result = response.get("result", "")
    if not result:
        return "No graph results found."
    return result


# -----------------------------
# Hybrid Search Tool
# -----------------------------
@tool
def hybrid_search(query: str):
    """
    Hybrid retrieval tool that combines:
    1. Semantic vector search
    2. Graph relationship search
    3. Reranking

    Use this for:
    - Course recommendations
    - Resume-based matching
    - Skill-based exploration
    - Program discovery
    - Questions requiring both semantic understanding and graph relationships
    """
    vector_db = get_vector_main()
    graph = get_graph_main()

    llm_model = get_llm(
        st.session_state.model_name,
        st.session_state.model_api_key
    )

    # VECTOR SEARCH
    vector_docs = vector_db.similarity_search(query, k=15)
    vector_results = []
    for d in vector_docs:
        vector_results.append({
            "source": "vector",
            "content": d.page_content,
            "metadata": d.metadata
        })

    # GRAPH SEARCH
    cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
    cypher_qa = GraphCypherQAChain.from_llm(
        llm_model,
        graph=graph,
        verbose=False,
        cypher_prompt=cypher_prompt,
        top_k=10,
        validate_cypher=True,
        allow_dangerous_requests=True,
        return_direct=False
    )
    graph_response = cypher_qa.invoke({"query": query})
    graph_result = graph_response.get("result", "")
    graph_results = []
    if graph_result:
        graph_results.append({
            "source": "graph",
            "content": str(graph_result)
        })

    # COMBINE RESULTS
    combined_results = vector_results + graph_results
    if not combined_results:
        return "No relevant results found."

    # BM25 RERANKING
    documents = [r["content"] for r in combined_results]
    tokenized_docs = [doc.lower().split() for doc in documents ]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # CROSS-ENCODER RERANK
    rerank_pairs = [[query, doc]for doc in documents]
    try:
        rerank_scores = reranker.predict(rerank_pairs)
        final_scores = (0.7 * np.array(rerank_scores)+ 0.3 * np.array(bm25_scores))
    except Exception:
        final_scores = bm25_scores

    # SORT RESULTS
    ranked_indices = np.argsort(final_scores)[::-1]
    final_results = []
    for idx in ranked_indices[:10]:
        final_results.append({
            "score": float(final_scores[idx]),
            "source": combined_results[idx]["source"],
            "content": combined_results[idx]["content"],
            "metadata": combined_results[idx].get("metadata", {})
        })
    return json.dumps(final_results, indent=2)
