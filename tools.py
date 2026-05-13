# Importing Libraries
import json
import streamlit as st
from llm import get_llm
from langchain.tools import tool
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
    llm_model = get_llm(st.session_state.model_name, st.session_state["MODEL_API_KEY"])

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
