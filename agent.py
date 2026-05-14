# Importing Libraries
import os
import streamlit as st
from llm import get_llm
from utils import get_session_id
from tools import vector_search, graph_search, hybrid_search
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from helper import infer_school_context, extract_resume_keywords
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Memory
# -----------------------------
def get_memory(session_id):
    return Neo4jChatMessageHistory(
        session_id=session_id,
        url=os.getenv('LOG_NEO4J_URI'),
        username=os.getenv('LOG_NEO4J_USERNAME'),
        password=os.getenv('LOG_NEO4J_PASSWORD'),
        database=os.getenv('LOG_NEO4J_DATABASE')
    )


# -----------------------------
# Tools
# -----------------------------
tools = [hybrid_search, vector_search, graph_search]


# -----------------------------
# Agent Prompt
# -----------------------------
agent_prompt = PromptTemplate.from_template("""
You are an expert academic and career advisor for The University of Texas at Dallas (UTD).

Your role is to help users:
- Explore schools, programs, certificates, and courses.
- Understand professor expertise and course content.
- Identify relevant skills taught in courses.
- Recommend learning paths based on career goals.
- Match resume skills and interests to UTD offerings.

Use the provided student context and metadata below to personalize recommendations.

--------------------------------------------------
STUDENT PROFILE CONTEXT
--------------------------------------------------
Resume Context:
{resume_context}

Additional Extracted Context:
{additional_context}

--------------------------------------------------
PRIMARY RETRIEVAL STRATEGY
--------------------------------------------------
You have access to three retrieval tools:

1. hybrid_search
   - Combines semantic vector retrieval + graph relationship retrieval.
   - Best for MOST user questions.
   - Preferred default retrieval tool.

2. graph_search
   - Best for highly structured factual relationship queries.
   - Uses Neo4j graph traversal and Cypher generation.

3. vector_search
   - Best for pure semantic similarity or recommendation-only tasks.
   - Uses embeddings and semantic retrieval only.

--------------------------------------------------
TOOL SELECTION RULES
--------------------------------------------------

DEFAULT BEHAVIOR:
- Prefer 'hybrid_search' for most queries because it combines:
  - semantic understanding
  - relationship reasoning
  - resume matching
  - recommendation quality

Use 'hybrid_search' when:
- Recommending courses or programs
- Resume-based matching
- Career-path discovery
- Skill-to-course matching
- Multi-step exploration
- Questions involving both meaning and relationships
- General academic advising

--------------------------------------------------
USE graph_search ONLY WHEN:
--------------------------------------------------
- The user asks for highly specific factual graph relationships.
- The query depends on exact graph structure.

Examples:
- "Who teaches CS 6360?"
- "What are the prerequisites for MIS 6382?"
- "Which professors teach machine learning?"
- "Which skills are taught in BUAN 6359?"
- "Which courses belong to the MS Business Analytics program?"

Graph queries are best for:
- prerequisites
- instructors
- degree requirements
- direct relationships
- exact course/program mappings

--------------------------------------------------
USE vector_search ONLY WHEN:
--------------------------------------------------
- The question is purely semantic and does NOT require graph relationships.
- The user wants broad recommendations or similarity matching.

Examples:
- "Recommend courses for data science"
- "Find programs matching my resume"
- "Suggest AI-related electives"
- "What courses align with cloud computing?"

--------------------------------------------------
MULTI-TOOL STRATEGY
--------------------------------------------------
If needed:
1. Use hybrid_search first.
2. Then refine with graph_search if exact factual details are needed.
3. Use vector_search only as a fallback semantic retrieval method.

Do NOT call multiple tools unnecessarily.

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------
- Never fabricate relationships or course details.
- Never assume graph schema structure.
- Only use information returned by tools.
- Personalize recommendations using the resume context whenever available.
- Be concise, informative, and specific to UTD.
- If tool results are insufficient, clearly say so instead of guessing.

--------------------------------------------------
TOOLS
--------------------------------------------------

You have access to the following tools:

{tools}

To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When responding directly to the user:

Thought: Do I need to use a tool? No
Final Answer: your response here

--------------------------------------------------
CONVERSATION HISTORY
--------------------------------------------------
{chat_history}

--------------------------------------------------
NEW INPUT
--------------------------------------------------
{input}

{agent_scratchpad}
""")


# -----------------------------
# Create Chat Agent
# -----------------------------
def build_chat_agent():
    llm_model = get_llm(
        st.session_state.get("model_name"),
        st.session_state.get("model_api_key")
    )

    agent = create_react_agent(
        llm=llm_model,
        tools=tools,
        prompt=agent_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10
    )

    chat_agent  = RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chat_agent


# -----------------------------
# Generation Function
# -----------------------------
def generate_response(user_input):
    # GET ACTIVE LLM
    active_llm = get_llm(
        st.session_state.get("model_name"),
        st.session_state.get("model_api_key")
    )

    # SKILL INFERENCE
    keywords = extract_resume_keywords(active_llm)
    keyword_summary = ", ".join(keywords) if keywords else "No specific skills inferred."
    st.session_state.additional_data = st.session_state.additional_data + "\n Inferred Skills: \n" + keyword_summary

    # SCHOOL INFERENCE
    school_summary = infer_school_context(keywords)
    if school_summary != "No specific school details inferred.":
        st.session_state.additional_data = st.session_state.additional_data + "\n Inferred School: \n" + school_summary

    live_chat_agent = build_chat_agent()
    response = live_chat_agent.invoke(
        {
            "input": user_input,
            "resume_context": st.session_state.get("resume_data") or "No resume uploaded yet.",
            "additional_context": st.session_state.get("additional_data") or "No additional context inferred.",
        },
        config={
            "configurable": {
                "session_id": get_session_id()
            }
        }
    )
    return response["output"]
