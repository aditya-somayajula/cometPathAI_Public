# Importing Libraries
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from langchain_community.document_loaders import PyPDFLoader


# -----------------------------
# Write to UI
# -----------------------------
def write_message(role, content, save = True):
    if save:
        st.session_state.messages.append({'role': role, 'content': content})
    with st.chat_message(role):
        st.markdown(content)


# -----------------------------
# Get session ID
# -----------------------------
def get_session_id():
    return get_script_run_ctx().session_id


# -----------------------------
# Document Analyzer
# -----------------------------
def get_resume(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n\n---- THIS IS CUSTOM PAGE ENDING -----\n\n".join([p.page_content for p in pages])
    st.session_state.resume_data = full_text
    st.session_state.resume_upload = True
