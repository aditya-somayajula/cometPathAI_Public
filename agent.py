# Importing Libraries
import streamlit as st
from llm import get_llm
from utils import get_session_id
from prompts import get_generation_system_prompt
from tools import vector_search, graph_search
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------
# Memory
# -----------------------------
def get_memory(session_id):
    return Neo4jChatMessageHistory(
        session_id=session_id,
        url=st.session_state.LOG_NEO4J_URI,
        username=st.session_state.LOG_NEO4J_USERNAME,
        password=st.session_state.LOG_NEO4J_PASSWORD,
        database=st.session_state.LOG_NEO4J_DATABASE
    )


# -----------------------------
# Tools
# -----------------------------
tools = [vector_search, graph_search]


# -----------------------------
# Agent Prompt
# -----------------------------
agent_prompt = PromptTemplate.from_template("""
{system_prompt}

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
        st.session_state.get("MODEL_API_KEY")
    )

    agent = create_react_agent(
        llm=llm_model,
        tools=tools,
        prompt=agent_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    chat_agent = RunnableWithMessageHistory(
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
    chat_agent = build_chat_agent()

    system_prompt = get_generation_system_prompt({
        "resume_context": st.session_state.get(
            "resume_data"
        ) or ""
    })
    response = chat_agent.invoke(
        {
            "input": user_input,
            "system_prompt": system_prompt
        },
        config={
            "configurable": {
                "session_id": get_session_id()
            }
        }
    )
    return response["output"]
