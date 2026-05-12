# Importing Libraries
import tempfile
import streamlit as st
from datetime import datetime
from utils import write_message, get_resume, get_session_id
from agent import generate_response


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config('CometPathAI', page_icon=':man_student:', layout='wide')
if 'resume_upload' not in st.session_state:
    st.session_state.resume_upload = False
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None

st.header("CometPathAI 👨‍🎓", text_alignment="center", divider="gray")
with st.sidebar:

    # Display Session ID
    with st.expander("**Session Details**", icon="ℹ️"):
        st.info(f"""
            * Session ID: {get_session_id()}
            * Start time: {datetime.now()}
        """)

    # Display Usage
    with st.expander("**Usage**", icon="ℹ️"):
        st.info("""
            1.  **Upload your resume:** Begin by uploading your resume file. The chatbot will process your document to extract key information.
            2.  **Wait for the analysis:** The AI will analyze your skills, experience, and keywords to understand your professional profile. This may take a moment.
            3.  **Review your matches:** Once the analysis is complete, the chatbot will provide a list of potential matches. You can click on each one to view more details.
            4.  **Refine your search:** If the initial matches aren't what you're looking for, you can provide more information or clarify your preferences in the chat to get more relevant results.
            """)

    # Display Disclaimer
    with st.expander("**Disclaimer**", icon="⚠️"):
        st.warning("""
            This AI chatbot provides course/programs matches based on the information you provide in your resume. The matches are generated using an algorithm that analyzes keywords, skills, and experience. Please be aware of the following:
            * **Accuracy is not guaranteed:** The matches are suggestions and may not be a perfect fit. We recommend you review each suggestion carefully to determine if it aligns with your career goals and qualifications.
            * **Data privacy:** We prioritize your privacy. The resume data you submit is used solely for the purpose of generating matches and is not shared with third parties.
            * **Bias:** While we strive to be impartial, the AI's suggestions may reflect biases present in the training data. We encourage you to explore a variety of sources in your search.
            * **Not a substitute for professional advice:** This tool is a supplement to your search, not a replacement for professional education counseling.
            """, )

    # For file upload
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader("Choose a file", type="pdf", key=f"resume_uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Get Resume Data
        with st.spinner('In Progress...'):
            get_resume(temp_file_path)

        if "resume_data" in st.session_state:
            if st.session_state.resume_data is not None:
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        else:
            st.error("Unable to upload the file. Please try again")

    # For Chat Refresh
    if st.sidebar.button("Refresh Chat"):
        st.session_state.uploader_key += 1
        st.session_state.resume_upload = False
        st.session_state.resume_data = None
        st.session_state.messages = [
            {'role': 'assistant',
             'content': "Hi there! Ready to find your next program? Upload your resume, and I'll instantly "
                        "match your skills with various degree/certificate programs at University of Texas at Dallas. Let's start the search!"},
        ]
        st.rerun()


# -----------------------------
# Set up Session State
# -----------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant',
         'content': "Hi there! Ready to find your next program? Upload your resume, and I'll instantly "
                    "match your skills with various degree/certificate programs at University of Texas at Dallas. Let's start the search!"},
    ]


# -----------------------------
# Submit handler
# -----------------------------
def handle_submit(user_message):
    with st.spinner('Thinking...'):
        response = generate_response(user_message)
        write_message('assistant', response)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input('What is up?'):
    write_message('user', prompt)
    handle_submit(prompt)
