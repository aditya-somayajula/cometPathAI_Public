# Importing Libraries
import tempfile
import streamlit as st
from agent import generate_response
from utils import get_resume, write_message
from llm import DEFAULT_MODEL, MODEL_OPTIONS
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Page Config, Initialization, Global Variables
# -----------------------------
st.set_page_config("CometPathAI", page_icon=":mortar_board:", layout="wide")

required_keys = [
    "MAIN_NEO4J_URI", "MAIN_NEO4J_USERNAME", "MAIN_NEO4J_PASSWORD", "MAIN_NEO4J_DATABASE",
    "LOG_NEO4J_URI", "LOG_NEO4J_USERNAME", "LOG_NEO4J_PASSWORD", "LOG_NEO4J_DATABASE",
    "COLLECTION_NAME", "CHROMA_TENANT", "CHROMA_API_KEY", "CHROMA_DB",
    "HF_TOKEN", "LANGSMITH_API_KEY", "MODEL_API_KEY"
]

WELCOME_MESSAGE = (
    "Hi there! Upload your resume, and I'll match your skills with graduate "
    "programs and courses at University of Texas at Dallas."
)

SUGGESTED_PROMPTS = [
    "Which UTD graduate courses match my resume?",
    "Who teaches MIS 6382 and when is it offered?",
    "Recommend programs for a data analytics career.",
    "Which courses should I take for Python and AI skills?",
]

def initialize_session_state():
    defaults = {
        "resume_upload": False,
        "resume_data": None,
        "uploader_key": 0,
        "model_name": DEFAULT_MODEL,
        "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for kys in required_keys:
        st.session_state[kys] = ""

initialize_session_state()


# -----------------------------
# Define Functions
# -----------------------------
def apply_theme():
    st.markdown(
        """
        <style>
            :root {
                --cp-bg: #f6f8fb;
                --cp-panel: #ffffff;
                --cp-text: #172033;
                --cp-muted: #667085;
                --cp-line: #d9e2ef;
                --cp-blue: #2454d6;
                --cp-green: #0f8f6f;
                --cp-gold: #c78b12;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(36, 84, 214, 0.10), transparent 34rem),
                    linear-gradient(180deg, #f9fbff 0%, var(--cp-bg) 100%);
                color: var(--cp-text);
            }

            .block-container {
                max-width: 1120px;
                padding-top: 2rem;
                padding-bottom: 5.5rem;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--cp-line);
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1.4rem;
            }

            .cp-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
                border: 1px solid var(--cp-line);
                border-radius: 18px;
                padding: 1.5rem 1.6rem;
                box-shadow: 0 18px 45px rgba(23, 32, 51, 0.08);
                margin-bottom: 1.15rem;
            }

            .cp-kicker {
                color: var(--cp-blue);
                font-size: 0.78rem;
                font-weight: 750;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .cp-title {
                color: var(--cp-text);
                font-size: clamp(2rem, 4vw, 3.15rem);
                line-height: 1.04;
                font-weight: 800;
                margin: 0;
                letter-spacing: 0;
            }

            .cp-subtitle {
                color: var(--cp-muted);
                font-size: 1rem;
                max-width: 720px;
                margin: 0.7rem 0 0;
            }

            .cp-status-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 1rem 0 1.25rem;
            }

            .cp-status-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--cp-line);
                border-radius: 12px;
                padding: 0.9rem 1rem;
            }

            .cp-status-label {
                color: var(--cp-muted);
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            .cp-status-value {
                color: var(--cp-text);
                font-size: 1rem;
                font-weight: 750;
                margin-top: 0.25rem;
            }

            .cp-section-label {
                color: var(--cp-muted);
                font-size: 0.85rem;
                font-weight: 700;
                margin: 0.4rem 0 0.25rem;
            }

            .cp-sidebar-title {
                font-size: 1.3rem;
                font-weight: 800;
                color: var(--cp-text);
                margin-bottom: 0.2rem;
            }

            .cp-sidebar-note {
                color: var(--cp-muted);
                font-size: 0.9rem;
                margin-bottom: 1rem;
            }

            .stButton > button {
                border-radius: 10px;
                border: 1px solid var(--cp-line);
                background: #ffffff;
                color: var(--cp-text);
                font-weight: 650;
                min-height: 2.5rem;
            }

            .stButton > button:hover {
                border-color: var(--cp-blue);
                color: var(--cp-blue);
                background: #f5f8ff;
            }

            [data-testid="stFileUploader"] {
                background: #f8fbff;
                border: 1px dashed #a9b9d6;
                border-radius: 14px;
                padding: 0.85rem;
            }

            [data-testid="stChatMessage"] {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(217, 226, 239, 0.9);
                border-radius: 14px;
                padding: 0.35rem 0.6rem;
                box-shadow: 0 8px 24px rgba(23, 32, 51, 0.04);
            }

            [data-testid="stChatInput"] {
                background: rgba(255, 255, 255, 0.92);
                border-top: 1px solid var(--cp-line);
            }

            @media (max-width: 760px) {
                .cp-status-row {
                    grid-template-columns: 1fr;
                }

                .cp-hero {
                    padding: 1.2rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_chat():
    st.session_state.uploader_key += 1
    st.session_state.resume_upload = False
    st.session_state.resume_data = None
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.rerun()


def render_hero():
    resume_status = "Uploaded" if st.session_state.resume_data else "Not uploaded"
    chat_turns = max(len(st.session_state.messages) - 1, 0)

    st.markdown(
        f"""
        <section class="cp-hero">
            <div class="cp-kicker">UT Dallas graduate advising assistant</div>
            <h1 class="cp-title">CometPathAI</h1>
            <p class="cp-subtitle">
                Upload a resume, ask about UTD programs, compare graduate courses,
                and look up instructor or schedule details from your course data.
            </p>
        </section>
        <div class="cp-status-row">
            <div class="cp-status-card">
                <div class="cp-status-label">Resume</div>
                <div class="cp-status-value">{resume_status}</div>
            </div>
            <div class="cp-status-card">
                <div class="cp-status-label">Knowledge Base</div>
                <div class="cp-status-value">Courses, programs, schedules</div>
            </div>
            <div class="cp-status-card">
                <div class="cp-status-label">Conversation</div>
                <div class="cp-status-value">{chat_turns} messages</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="cp-sidebar-title">CometPathAI</div>
            <div class="cp-sidebar-note">
                Resume-aware graduate course and program matching for UTD.
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_list = list(MODEL_OPTIONS)
        if st.session_state.model_name not in model_list:
            st.session_state.model_name = DEFAULT_MODEL
        selected_index = model_list.index(st.session_state.model_name)
        selected_model_label = st.selectbox(
            "Response model",
            options=model_list,
            index=selected_index
        )
        st.session_state.model_name = selected_model_label
        st.caption(f"Active model: {st.session_state.model_name}")

        col1, col2, col3 = st.columns([2, 2, 2], vertical_alignment="center")
        with col1:
            with st.popover(label="Main Graph", help="Main Graph DB Details"):
                st.session_state.MAIN_NEO4J_URI = st.text_input("URI (Main)", value=st.session_state.get("MAIN_NEO4J_URI", ""))
                st.session_state.MAIN_NEO4J_USERNAME = st.text_input("Username (Main)", value=st.session_state.get("MAIN_NEO4J_USERNAME",""))
                st.session_state.MAIN_NEO4J_PASSWORD = st.text_input("Password (Main)", type="password", value=st.session_state.get("MAIN_NEO4J_PASSWORD",""))
                st.session_state.MAIN_NEO4J_DATABASE = st.text_input("Database (Main)", value=st.session_state.get("MAIN_NEO4J_DATABASE",""))
        with col2:
            with st.popover(label="Log Graph", help="Log Graph DB Details"):
                st.session_state.LOG_NEO4J_URI = st.text_input("URI (Log)", value=st.session_state.get("LOG_NEO4J_URI", ""))
                st.session_state.LOG_NEO4J_USERNAME = st.text_input("Username (Log)", value=st.session_state.get("LOG_NEO4J_USERNAME",""))
                st.session_state.LOG_NEO4J_PASSWORD = st.text_input("Password (Log)", type="password", value=st.session_state.get("LOG_NEO4J_PASSWORD",""))
                st.session_state.LOG_NEO4J_DATABASE = st.text_input("Database (Log)", value=st.session_state.get("LOG_NEO4J_DATABASE",""))
        with col3:
            with st.popover(label="Vector DB", help="Vector DB Details"):
                st.session_state.COLLECTION_NAME = st.text_input("Collection Name",value=st.session_state.get("COLLECTION_NAME", ""))
                st.session_state.CHROMA_TENANT = st.text_input("Tenant", value=st.session_state.get("CHROMA_TENANT", ""))
                st.session_state.CHROMA_API_KEY = st.text_input("API Key", value=st.session_state.get("CHROMA_API_KEY",""))
                st.session_state.CHROMA_DB = st.text_input("Database", value=st.session_state.get("CHROMA_DB",""))

        col4, col5, col6 = st.columns([5, 5, 5], vertical_alignment="center")
        with col5:
            with st.popover(label="API Details", help="API Details"):
                st.session_state.HF_TOKEN = st.text_input("Hugging Face Token", type="password", value=st.session_state.get("HF_TOKEN", ""))
                st.session_state.LANGSMITH_API_KEY = st.text_input("Langsmith API", type="password", value=st.session_state.get("LANGSMITH_API_KEY",""))
                st.session_state.MODEL_API_KEY = st.text_input("Model API", type="password", value=st.session_state.get("MODEL_API_KEY",""))

        st.divider()

        uploaded_file = st.file_uploader(
            "Upload resume PDF",
            type="pdf",
            key=f"resume_uploader_{st.session_state.uploader_key}",
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            with st.spinner("Analyzing resume..."):
                get_resume(temp_file_path)

            if st.session_state.resume_data is not None:
                st.success(f"{uploaded_file.name} uploaded successfully.")
            else:
                st.error("Unable to upload the file. Please try again.")

        st.divider()

        if st.button("Refresh chat", use_container_width=True):
            reset_chat()

        with st.expander("Disclaimer"):
            st.warning(
                """
                Matches are suggestions and may not be a perfect fit. Review each
                course or program against your career goals and official UTD
                requirements. Resume data is used only to generate matches.
                """
            )

        with st.expander("How to use"):
            st.info(
                """
                1. Upload your resume PDF.
                2. Ask for course or program recommendations.
                3. Ask follow-up questions about prerequisites, instructors, or schedules.
                4. Refine the answer with your preferences.
                """
            )


def render_suggested_prompts():
    st.markdown('<div class="cp-section-label">Try a question</div>', unsafe_allow_html=True)
    columns = st.columns(2)
    selected_prompt = None
    for index, prompt in enumerate(SUGGESTED_PROMPTS):
        with columns[index % 2]:
            if st.button(prompt, key=f"suggested_prompt_{index}", use_container_width=True):
                selected_prompt = prompt
    return selected_prompt


def handle_submit(user_message):
    missing_keys = [
        key for key in required_keys
        if key not in st.session_state or str(st.session_state[key]).strip() == "" or st.session_state[key] == None
    ]
    if missing_keys:
        st.error(
            f"The following required configuration values are missing or empty:\n\n"
            + "\n".join([f"- {key}" for key in missing_keys])
        )
        st.stop()
    with st.spinner("Thinking..."):
        response = generate_response(user_message)
        write_message("assistant", response)


# -----------------------------
# Call Functions
# -----------------------------
apply_theme()
render_sidebar()
render_hero()

selected_prompt = render_suggested_prompts()
chat_prompt = st.chat_input("Ask about programs, courses, instructors, or schedules")

st.divider()
for message in st.session_state.messages:
    write_message(message["role"], message["content"], save=False)

prompt = selected_prompt or chat_prompt
if prompt:
    write_message("user", prompt)
    handle_submit(prompt)
