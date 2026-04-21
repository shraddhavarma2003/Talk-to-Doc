import streamlit as st
import os
import tempfile
import traceback
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

# --- Backend Configuration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "talk-to-doc")

# Modular imports
from src.ingestion import process_multiple_documents
from src.retrieval import create_vectorstore, get_embeddings, get_vectorstore
from src.llm import get_rag_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="Talk to Doc | AI Knowledge Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Production-Grade UI & Accessibility ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Base Font & Global Contrast */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Content Background - Light Theme */
    .stApp {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }

    /* Sidebar - Dark Theme */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    /* Global Text Visibility - Main Content */
    .main p, .main li, .main h1, .main h2, .main h3, .main span, .main label, .stMarkdown {
        color: #111111 !important;
    }

    /* Chat Messages Container */
    [data-testid="stChatMessageContainer"] {
        background-color: transparent !important;
    }

    /* Chat Messages Base Styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
    }

    /* --- User Message Bubble --- */
    [data-testid="stChatMessage"][aria-label="Chat message from user"],
    .st-emotion-cache-1c7n2ka { /* Fallback for user message container */
        background-color: #f1f3f5 !important;
        border: 1px solid #e9ecef !important;
        color: #111111 !important;
    }
    
    [data-testid="stChatMessage"][aria-label="Chat message from user"] p,
    [data-testid="stChatMessage"][aria-label="Chat message from user"] span,
    [data-testid="stChatMessage"][aria-label="Chat message from user"] div {
        color: #111111 !important;
    }

    /* --- Assistant Message Bubble --- */
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"],
    .st-emotion-cache-10pw50 { /* Fallback for assistant message container */
        background-color: #ffffff !important;
        border: 1px solid #e9ecef !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        color: #111111 !important;
    }

    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] p,
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] span,
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] div {
        color: #111111 !important;
    }

    /* Thinking / Spinner Visibility */
    .stSpinner > div > div {
        border-color: #111111 !important;
    }
    .stSpinner p {
        color: #111111 !important;
        font-weight: 500 !important;
    }

    /* Chat Input Fixes */
    .stChatInputContainer {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stChatInputContainer textarea {
        color: #111111 !important;
    }

    /* Success/Error/Info Alert Text */
    .stAlert p {
        color: #111111 !important;
    }

    /* Card Styling */
    .view-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        color: #111111;
    }

    /* Upload Area Visibility */
    .stFileUploader {
        border-radius: 16px;
        background-color: #ffffff !important;
        padding: 20px;
        border: 2px dashed #cbd5e1;
    }
    .stFileUploader label, .stFileUploader p {
        color: #111111 !important;
    }

    /* Expander Visibility Fix */
    .stExpander {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        margin-top: 0.5rem !important;
    }
    .stExpander summary label p, 
    .stExpander summary p, 
    .stExpander [data-testid="stExpanderDetails"] p,
    .stExpander [data-testid="stExpanderDetails"] span {
        color: #111111 !important;
        font-weight: 500 !important;
    }
    .stExpander svg {
        fill: #111111 !important;
    }

    /* Button Styling */
    .stButton button {
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = [] # List of dicts with file metadata
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Helper Functions ---
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return "Online" if response.status_code == 200 else "Offline"
    except:
        return "Offline"

def check_pinecone():
    try:
        if not PINECONE_API_KEY: return "Offline"
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()
        return "Online"
    except:
        return "Offline"

def validate_config():
    if not PINECONE_API_KEY or "your_" in PINECONE_API_KEY.lower():
        st.sidebar.error("⚠️ Config Missing: PINECONE_API_KEY")
        return False
    return True

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=60)
    st.title("Talk to Doc")
    st.markdown("Your AI Knowledge Partner")
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📤 Upload Documents", "📚 Document Library", "💬 Conversation", "📊 System Status"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("### 🛠 Tools")
    if st.button("🗑️ Reset All Session Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Main Page Routing ---
is_config_valid = validate_config()

# Pre-load cached vectorstore if config is valid
if is_config_valid and not st.session_state.vectorstore:
    try:
        embeddings = get_embeddings()
        st.session_state.vectorstore = get_vectorstore(embeddings, PINECONE_API_KEY, PINECONE_INDEX)
        # Note: vectorstore_ready is still false until at least one file is processed/detected
    except:
        pass

if not is_config_valid:
    st.warning("### ⚠️ Application Not Configured")
    st.info("Please set your `PINECONE_API_KEY` in the `.env` file to begin.")
    st.stop()

if page == "📤 Upload Documents":
    st.markdown('<div class="main-title">Upload Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Add your PDF or Word documents to the knowledge base.</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX", 
            type=["pdf", "docx"], 
            accept_multiple_files=True,
            help="Your files will be automatically chunked and indexed into Pinecone."
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in [x['name'] for x in st.session_state.processed_files]]
            
            if new_files:
                if st.button(f"Process {len(new_files)} New Documents", type="primary"):
                    with st.status("🚀 Indexing Knowledge Base...", expanded=True) as status:
                        temp_paths = []
                        original_names = [file.name for file in new_files]
                        try:
                            for uploaded_file in new_files:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_paths.append(tmp_file.name)
                            
                            # Backend processing
                            progress_bar = st.progress(0, text="Indexing starting...")
                            all_splits, stats = process_multiple_documents(temp_paths, original_names)
                            
                            progress_bar.progress(50, text="Uploading to Pinecone...")
                            create_vectorstore(all_splits, PINECONE_API_KEY, PINECONE_INDEX)
                            
                            # Refresh cached vectorstore connection
                            embeddings = get_embeddings()
                            st.session_state.vectorstore = get_vectorstore(embeddings, PINECONE_API_KEY, PINECONE_INDEX)
                            
                            progress_bar.progress(100, text="Indexing complete!")
                            st.session_state.vectorstore_ready = True
                            
                            # Store metadata
                            for f in new_files:
                                st.session_state.processed_files.append({
                                    "name": f.name,
                                    "size": f"{f.size / 1024:.1f} KB",
                                    "type": f.type,
                                    "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                                })
                            
                            status.update(label="✅ Knowledge Base Updated!", state="complete")
                            st.balloons()
                            st.success(f"Successfully indexed {len(new_files)} files into {PINECONE_INDEX}.")
                        except Exception as e:
                            status.update(label="❌ Ingestion Failed", state="error")
                            st.error(f"Error: {str(e)}")
                            with st.expander("Traceback"):
                                st.code(traceback.format_exc())
                        finally:
                            for path in temp_paths:
                                if os.path.exists(path): os.remove(path)
            else:
                st.info("All uploaded files are already in the knowledge base.")

elif page == "📚 Document Library":
    st.markdown('<div class="main-title">Document Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">View and manage your indexed documents.</div>', unsafe_allow_html=True)
    
    if not st.session_state.processed_files:
        st.info("No documents have been indexed yet. Head over to the Upload section!")
    else:
        # Display as a clean table
        st.dataframe(
            st.session_state.processed_files,
            column_config={
                "name": "File Name",
                "size": "File Size",
                "type": "MIME Type",
                "date": "Upload Date"
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("### 🔍 Document Previews")
        for i, doc in enumerate(st.session_state.processed_files):
            with st.expander(f"View details for {doc['name']}"):
                col1, col2 = st.columns(2)
                col1.metric("Status", "Indexed")
                col1.write(f"**Type:** {doc['type']}")
                col2.write(f"**Indexed on:** {doc['date']}")
                col2.write(f"**File Size:** {doc['size']}")

elif page == "💬 Conversation":
    st.markdown('<div class="main-title">Conversation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ask questions about your indexed documents.</div>', unsafe_allow_html=True)
    
    if not st.session_state.vectorstore_ready:
        st.warning("⚠️ Knowledge base is empty. Please upload documents first.")
        st.stop()

    # Chat UI
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("🤖 Thinking..."):
                        chain, retriever = get_rag_chain(st.session_state.vectorstore)
                        docs = retriever.invoke(prompt)
                        answer = chain.invoke(prompt)
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        with st.expander("📌 Reference Context"):
                            for i, doc in enumerate(docs):
                                src = doc.metadata.get("source", "Unknown")
                                pg = doc.metadata.get("page", "N/A")
                                st.markdown(f"**[{i+1}] {src}** (pg. {pg})")
                                st.caption(doc.page_content[:300] + "...")
                                if i < len(docs) - 1: st.divider()
                except Exception as e:
                    st.error(f"Chat Error: {str(e)}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

elif page == "📊 System Status":
    st.markdown('<div class="main-title">System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Monitor your local AI infrastructure health.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="view-card">', unsafe_allow_html=True)
        st.metric("Ollama Engine", check_ollama())
        st.caption("Local LLM Server (Llama 3)")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="view-card">', unsafe_allow_html=True)
        st.metric("Pinecone Index", check_pinecone())
        st.caption(f"Vector Database: {PINECONE_INDEX}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="view-card">', unsafe_allow_html=True)
        st.metric("Indexed Files", len(st.session_state.processed_files))
        st.caption("Active Knowledge Base")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🛠 Infrastructure Details")
    st.json({
        "Model": "llama3 (via Ollama)",
        "Embeddings": "sentence-transformers/all-MiniLM-L6-v2",
        "Framework": "LangChain 0.3",
        "Vector Provider": "Pinecone Serverless"
    })
