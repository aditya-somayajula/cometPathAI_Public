# Importing Libraries
import os
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
    ctx = get_script_run_ctx()
    if ctx is None:
        return "default-session"
    return ctx.session_id


# -----------------------------
# Document Analyzer
# -----------------------------
def get_resume(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n\n---- PAGE ENDING -----\n\n".join([p.page_content for p in pages])
        st.session_state.resume_data = full_text
        st.session_state.resume_upload = True
        os.remove(file_path)
    except Exception as e:
        st.session_state.resume_data = None
        st.error(f"Resume processing failed: {str(e)}")
