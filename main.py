# ==========================
# Import Required Libraries
# ==========================

import os  # Used to access environment variables
import streamlit as st  # Streamlit UI framework
from dotenv import load_dotenv  # Load .env variables
from PyPDF2 import PdfReader  # Read PDF files

# Vector database
from langchain_community.vectorstores import FAISS

# LangChain message types for conversation memory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# Gemini LLM + Gemini Embeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# Split large text into chunks
from langchain_text_splitters import CharacterTextSplitter


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# Read Gemini API Key from .env file
api_key = os.getenv("GOOGLE_API_KEY")


# ==========================
# Streamlit Page Config
# ==========================

st.set_page_config(
    page_title="AI Assistant Hub",
    page_icon="🤖",
)

st.markdown("# 🤖 AI Assistant Hub")
st.caption(
    "Use the general chat or upload a PDF to ask questions about its content."
)


# ==========================
# Create Tabs
# ==========================

chat_tab, pdf_tab = st.tabs(
    ["General Chat", "PDF Q&A"]
)


# =====================================================
#                 GENERAL CHAT TAB
# =====================================================

with chat_tab:

    st.markdown("### General Chat")

    # Create two columns for buttons
    col1, col2 = st.columns([1, 1])

    # --------------------------
    # Thinking Mode Button
    # --------------------------

    with col1:

        thinking_mode = st.button(
            "🧠 Thinking mode"
            if not st.session_state.get(
                "thinking_mode",
                False
            )
            else "🧠 Thinking mode (ON)",
            use_container_width=True,
        )

    # --------------------------
    # New Chat Button
    # --------------------------

    with col2:

        if st.button(
            "🔄 New chat",
            use_container_width=True,
        ):

            # Remove old chat history
            st.session_state.pop(
                "flowmessage",
                None
            )

            st.session_state.pop(
                "messages",
                None
            )

            st.session_state[
                "thinking_mode"
            ] = False

            st.rerun()

    # Toggle Thinking Mode
    if thinking_mode:

        st.session_state[
            "thinking_mode"
        ] = not st.session_state.get(
            "thinking_mode",
            False
        )

    # --------------------------
    # Initialize LLM Memory
    # --------------------------

    if "flowmessage" not in st.session_state:

        st.session_state[
            "flowmessage"
        ] = [

            # System Prompt
            SystemMessage(
                content=
                "You are a helpful assistant. "
                "Answer clearly and concisely."
            )
        ]

    # --------------------------
    # Initialize UI Messages
    # --------------------------

    if "messages" not in st.session_state:

        st.session_state[
            "messages"
        ] = [
            {
                "role": "assistant",
                "content":
                "Hi! I'm ready to help. "
                "Ask me anything."
            }
        ]

    # ==========================
    # Gemini Response Function
    # ==========================

    def get_gemini_response(query):

        # Store User Message
        st.session_state[
            "flowmessage"
        ].append(
            HumanMessage(
                content=query
            )
        )

        # Check Thinking Mode
        thinking_enabled = (
            st.session_state.get(
                "thinking_mode",
                False
            )
        )

        # Higher temperature = more creative
        temperature = (
            0.8
            if thinking_enabled
            else 0.2
        )

        # Gemini Model
        response = (
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=temperature,
            )
            .invoke(
                st.session_state[
                    "flowmessage"
                ]
            )
        )

        # Extract Text
        answer = (
            response.content
            if hasattr(
                response,
                "content"
            )
            else str(response)
        )

        # Store AI Response
        st.session_state[
            "flowmessage"
        ].append(
            AIMessage(
                content=answer
            )
        )

        return answer

    # ==========================
    # Display Chat History
    # ==========================

    for message in st.session_state[
        "messages"
    ]:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # ==========================
    # Chat Input Box
    # ==========================

    user_input = st.chat_input(
        "Ask me anything...",
        key="chat_input"
    )

    if user_input:

        # Save User Message
        st.session_state[
            "messages"
        ].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # Show Loading Spinner
        with st.spinner(
            "Thinking..."
            if st.session_state.get(
                "thinking_mode",
                False
            )
            else "Generating answer..."
        ):

            answer = (
                get_gemini_response(
                    user_input
                )
            )

        # Save Assistant Response
        st.session_state[
            "messages"
        ].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.rerun()


# =====================================================
#                    PDF Q&A TAB
# =====================================================

with pdf_tab:

    st.markdown(
        "### PDF Question Answering"
    )

    st.write(
        "Upload a PDF and ask a "
        "question about its contents."
    )

    # --------------------------
    # Upload + Query Section
    # --------------------------

    upload_col, prompt_col = (
        st.columns([1.1, 1.9])
    )

    with upload_col:

        pdf = st.file_uploader(
            "Upload your PDF",
            type=["pdf"],
            key="pdf_uploader",
        )

    with prompt_col:

        query = st.text_input(
            "Ask a question about your PDF:",
            key="pdf_query",
        )

    # --------------------------
    # Process PDF
    # --------------------------

    if pdf is not None:

        try:

            # Read PDF
            pdf_reader = PdfReader(pdf)

            # Extract Text
            text = "".join(
                page.extract_text() or ""
                for page
                in pdf_reader.pages
            )

            # Empty PDF Check
            if not text.strip():

                st.warning(
                    "This PDF does not contain "
                    "extractable text."
                )

                st.stop()

            # --------------------------
            # Split Text into Chunks
            # --------------------------

            text_splitter = (
                CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
            )

            chunks = (
                text_splitter
                .split_text(text)
            )

            # --------------------------
            # Gemini Embeddings
            # --------------------------

            embeddings = (
                GoogleGenerativeAIEmbeddings(
                    model=
                    "models/gemini-embedding-001",
                    google_api_key=api_key,
                )
            )

            # Create FAISS Vector DB
            knowledge_base = (
                FAISS.from_texts(
                    chunks,
                    embeddings,
                )
            )

            # --------------------------
            # User Question
            # --------------------------

            if query:

                # Retrieve Top 4 Chunks
                docs = (
                    knowledge_base
                    .similarity_search(
                        query,
                        k=4,
                    )
                )

                # Build Context
                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                # Gemini Model
                llm = (
                    ChatGoogleGenerativeAI(
                        model=
                        "gemini-2.5-flash",
                        temperature=0.3,
                        google_api_key=api_key,
                    )
                )

                # Prompt
                prompt = f"""
You are a helpful assistant.

Answer using ONLY the PDF context.

Context:
{context}

Question:
{query}
"""

                # Generate Answer
                with st.spinner(
                    "Searching PDF..."
                ):

                    response = (
                        llm.invoke(
                            prompt
                        )
                    )

                # Display Answer
                st.write(
                    response.content
                )

        except Exception as exc:

            st.error(
                f"Error: {exc}"
            )

    else:

        st.info(
            "Upload a PDF to enable "
            "the PDF Q&A section."
        )