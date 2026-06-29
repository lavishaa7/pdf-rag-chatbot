import streamlit as st
from rag import load_and_split_pdf, get_vectorstore, get_conversational_chain, get_response
from langchain_core.callbacks import BaseCallbackHandler
import tempfile
import os
import uuid


class StreamHandler(BaseCallbackHandler):
    """Streams LLM tokens into a Streamlit placeholder as they arrive.

    The `container` and `text` are reset before each question, then
    `on_llm_new_token` updates the placeholder live as the final
    answer is generated.
    """

    def __init__(self):
        self.container = None
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        if self.container is not None:
            self.container.markdown(
                f'<div class="chat-bot">{self.text}</div>',
                unsafe_allow_html=True
            )

# Page config
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f0f4ff;
        }
        .main {
            background-color: #f0f4ff;
        }
        .stApp {
            background-color: #f0f4ff;
        }
        .title {
            text-align: center;
            color: #1a73e8;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .chat-user {
            background-color: #1a73e8;
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 75%;
            margin-left: auto;
            font-size: 0.95rem;
        }
        .chat-bot {
            background-color: #ffffff;
            color: #1a1a1a;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 75%;
            margin-right: auto;
            font-size: 0.95rem;
            border: 1px solid #d0d9f5;
        }
        .stTextInput > div > div > input {
            border: 2px solid #1a73e8;
            border-radius: 12px;
            padding: 10px 16px;
            font-size: 0.95rem;
        }
        .stButton > button {
            background-color: #1a73e8;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 0.95rem;
            font-weight: 600;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #1558b0;
        }
        .upload-section {
            background-color: #ffffff;
            border: 2px dashed #1a73e8;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .status-box {
            background-color: #e8f0fe;
            border-left: 4px solid #1a73e8;
            padding: 12px 16px;
            border-radius: 8px;
            color: #1a73e8;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #fdecea;
            border-left: 4px solid #d93025;
            padding: 12px 16px;
            border-radius: 8px;
            color: #d93025;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .source-box {
            background-color: #ffffff;
            color: #1a1a1a;
            border: 1px solid #d0d9f5;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        .source-box strong {
            color: #1a73e8;
        }
        /* Fix invisible label above input */
        .stTextInput label {
            color: #333333 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        /* Fix invisible expander header text */
        .streamlit-expanderHeader {
            color: #333333 !important;
            font-weight: 600 !important;
        }
        /* Fix expander arrow/icon color */
        .streamlit-expanderHeader svg {
            fill: #333333 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📄 PDF Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a PDF and chat with it using AI</div>', unsafe_allow_html=True)

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "stream_handler" not in st.session_state:
    st.session_state.stream_handler = StreamHandler()


def reset_session():
    """Clear everything so the user can upload a fresh PDF."""
    st.session_state.chain = None
    st.session_state.messages = []
    st.session_state.processed_file = None


# If a chatbot is already active, show "new PDF" and "clear chat" options
if st.session_state.chain is not None:
    st.markdown(
        f'<div class="status-box">✅ Currently chatting with: '
        f'<strong>{st.session_state.processed_file}</strong></div>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.session_state.chain.memory.clear()
            st.rerun()
    with col2:
        if st.button("📄 Upload a different PDF"):
            reset_session()
            st.rerun()
else:
    # File upload
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("Reading and processing your PDF..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                chunks = load_and_split_pdf(tmp_path)

                # Unique collection name per upload so vectors from
                # different PDFs/sessions never mix together
                collection_name = f"pdf_{uuid.uuid4().hex}"
                vectorstore = get_vectorstore(chunks, collection_name)
                st.session_state.chain = get_conversational_chain(
                    vectorstore, st.session_state.stream_handler
                )
                st.session_state.processed_file = uploaded_file.name

            except ValueError as e:
                # Expected, user-fixable issues (e.g. scanned PDF)
                st.markdown(f'<div class="error-box">⚠️ {e}</div>', unsafe_allow_html=True)

            except Exception as e:
                # Unexpected issues - most commonly Ollama not running
                # or the required models not pulled
                st.markdown(
                    '<div class="error-box">⚠️ Something went wrong while processing '
                    'your PDF. Make sure Ollama is running and that the '
                    '"llama3.2:1b" and "nomic-embed-text" models are pulled.</div>',
                    unsafe_allow_html=True
                )
                st.exception(e)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if st.session_state.chain is not None:
            st.rerun()

# Chat history display
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">{message["content"]}</div>', unsafe_allow_html=True)
        sources = message.get("sources")
        if sources:
            with st.expander("📎 Sources used for this answer"):
                for i, doc in enumerate(sources, start=1):
                    page = doc.metadata.get("page", "unknown")
                    snippet = doc.page_content[:200].strip().replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><strong>Source {i} (page {page}):</strong> '
                        f'{snippet}...</div>',
                        unsafe_allow_html=True
                    )

# Input
if st.session_state.chain is not None:
    question = st.text_input("Ask a question about your PDF", key=f"input_{st.session_state.input_key}")
    if st.button("Send"):
        if question.strip():
            st.session_state.messages.append({"role": "user", "content": question})
            st.markdown(f'<div class="chat-user">{question}</div>', unsafe_allow_html=True)

            placeholder = st.empty()
            placeholder.markdown('<div class="chat-bot">Thinking...</div>', unsafe_allow_html=True)

            handler = st.session_state.stream_handler
            handler.container = placeholder
            handler.text = ""

            try:
                answer, sources = get_response(st.session_state.chain, question)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ I couldn't generate a response. "
                                "Please check that Ollama is running and try again."
                })
                st.exception(e)
            finally:
                handler.container = None

            st.session_state.input_key += 1  # clears the input box
            st.rerun()
else:
    st.info("Please upload a PDF to start chatting.")