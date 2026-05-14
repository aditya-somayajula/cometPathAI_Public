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
# Hybrid Vector Search (Dense + Sparse) & Reranker
# -----------------------------
def hybrid_vector_search(docs, filter_type):
    try:
        if type(docs) != list:
            docs = [docs]

        print(filter_type)

        vector_store = get_vector_main()

        # Fetch Relevant Vector DB Records
        all_data = vector_store.similarity_search(query="", k=300, filter={"type": filter_type})
        all_documents = all_data["documents"]
        all_ids = all_data["ids"]
        all_metadata = all_data["metadatas"]
        print("Fetched all dta from vector store")

        # BM25 Tokenizer
        tokenized_docs = [
            doc.lower().split()
            for doc in all_documents
        ]
        bm25 = BM25Okapi(tokenized_docs)
        print("Tokenizer")

        all_candidates = []
        for doc in docs:
            # Vector Search
            vector_results = vector_store.similarity_search_with_score(query=doc, k=50, filter={"type": filter_type})

            for result_doc, score in vector_results:
                candidate = {
                    "id": result_doc.metadata.get("id"),
                    "text": result_doc.page_content,
                    "metadata": result_doc.metadata,
                    "distance": score
                }
                all_candidates.append(candidate)

            # BM25 retrival
            tokenized_query = doc.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[::-1][:5]
            for idx in top_indices:
                candidate = {
                    "id": all_ids[idx],
                    "text": all_documents[idx],
                    "metadata": all_metadata[idx],
                    "bm25_score": bm25_scores[idx]
                }
                all_candidates.append(candidate)
        print("Got all candidates")

        unique_candidates = {}
        for candidate in all_candidates:
            unique_candidates[candidate["id"]] = candidate
        candidate_docs = list(unique_candidates.values())
        print("Got unique candidates")

        combined_query = " ".join(docs)
        pairs = [
            (combined_query, doc["text"])
            for doc in candidate_docs
        ]
        print("Combined Query")

        # Re-rank
        scores = reranker.predict(pairs)
        reranked_results = sorted(
            zip(candidate_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        print("Rerannked")

        TOP_K = 10
        results_str = ""
        for rank, (doc, score) in enumerate(
                reranked_results[:TOP_K],
                start=1
        ):
            results_str += f"Rank #{rank} \n"
            results_str += f"Rerank Score: {score:.4f}\n"
            results_str += f"Content #{doc.get('text')} \n"
            results_str += f"Metadata #{doc.get('metadata')} \n"
            results_str += "\n\n ----- ----- ----- \n\n"
        print("Finished")

        return results_str
    except Exception as e:
        print(e)
        return "Unable to get results from vector database."
