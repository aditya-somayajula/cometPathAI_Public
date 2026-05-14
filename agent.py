# Importing Libraries
import os
import streamlit as st
from llm import get_llm
from utils import get_session_id
from tools import vector_search, graph_search
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from helper import extract_resume_keywords, infer_school_context
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
tools = [vector_search, graph_search]


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

Use the provided student context and metadata below to anchor your suggestions.

--------------------------------------------------
STUDENT PROFILE CONTEXT:
--------------------------------------------------
Resume Context: {resume_context}
Additional Extracted Context: {additional_context}

--------------------------------------------------
INSTRUCTIONS
--------------------------------------------------
Use graph_search when:
- Asking about relationships (teaches, includes, prerequisites).
- Asking about professors, courses, programs, skills.
- Asking structured factual questions.

Use vector_search when:
- Asking for recommendations.
- Asking for similar courses/programs.
- Resume-based matching.
- Skill-to-program discovery.

--------------------------------------------------
TOOL ROUTING RULES:
--------------------------------------------------
1. Always use 'graph_search' if the question involves:
   - Specific Course Codes (e.g., CS 6360, MIS 6382).
   - Faculty/Instructors (keywords: professor, who teaches, faculty, instructor).
   - Structural relations (prerequisites, degree requirements).

2. Always use 'vector_search' if the question involves:
   - Broad recommendations (e.g., "What should I take?").
   - Semantic similarity to the resume (e.g., "Match my Python skills").
   - Discovering programs for a specific career goal.

If a question requires both, call 'vector_search' first to find the course, then 'graph_search' to find the professor.

--------------------------------------------------
GENERIC DIRECTIONS
--------------------------------------------------
- Personalize recommendations using the user's background.
- Provide concise but informative responses.
- Be professional, encouraging, and specific to UTD.
- Never assume schema structure. Rely on tool output only. Never fabricate relationships.


TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: your response here
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}

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
        max_iterations=5,
        early_stopping_method="generate"
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
    school_summary = infer_school_context(st.session_state.resume_data, keywords)
    if school_summary != "No specific school details inferred.":
        st.session_state.additional_data = st.session_state.additional_data + "\n Recommended School: \n" + school_summary

    live_chat_agent = build_chat_agent()
    response = live_chat_agent.invoke(
        {
            "input": user_input,
            "resume_context": st.session_state.get("resume_data") or "No resume uploaded yet.",
            "additional_context": st.session_state.get("additional_data") or "No additional context calculated.",
        },
        config={
            "configurable": {
                "session_id": get_session_id()
            }
        }
    )
    return response["output"]
