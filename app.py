#!/usr/bin/env python3
"""
PDF RAG QA Streamlit Web App - Fixed Answer Display
Fixed the answer display formatting issue
"""

import streamlit as st
import tempfile
import os
from pdf_rag_qa import PDFRagQA
import time

# Set page configuration
st.set_page_config(
    page_title="PDF RAG QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for dark mode compatibility - FIXED
st.markdown("""
<style>
    /* Dark mode optimized styles */
    .main-header {
        text-align: center;
        color: #ffffff !important;
        padding: 2rem 0;
        border-bottom: 2px solid #444;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 1rem 0 2rem 0;
    }

    .main-header h1 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: #f0f0f0 !important;
        opacity: 0.9;
    }

    /* Upload section with dark mode support */
    .upload-section {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%) !important;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #68d391 !important;
        text-align: center;
        margin: 1rem 0;
        color: #ffffff !important;
    }

    .upload-section h3 {
        color: #68d391 !important;
    }

    /* Q&A section styling */
    .qa-section {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%) !important;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #4a5568;
        margin: 1rem 0;
        color: #ffffff !important;
    }

    .qa-section h3 {
        color: #68d391 !important;
    }

    /* FIXED Answer box with better contrast */
    .answer-box {
        background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #63b3ed !important;
        margin: 1rem 0;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        word-wrap: break-word;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    .answer-box strong {
        color: #e2e8f0 !important;
        font-size: 1.1em;
    }

    /* FIXED Source box styling */
    .source-box {
        background: linear-gradient(135deg, #2f855a 0%, #38a169 100%) !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #68d391 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        word-wrap: break-word;
    }

    .source-box strong {
        color: #e2e8f0 !important;
    }

    .source-box small {
        color: #cbd5e0 !important;
        opacity: 0.9;
        display: block;
        margin-top: 0.5rem;
    }

    /* Confidence indicators with better visibility */
    .confidence-high { 
        color: #68d391 !important; 
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .confidence-medium { 
        color: #f6e05e !important; 
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .confidence-low { 
        color: #fc8181 !important; 
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #68d391 !important;
        box-shadow: 0 0 0 2px rgba(104, 211, 145, 0.2) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(72, 187, 120, 0.4) !important;
    }

    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%) !important;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #68d391;
        margin: 0.5rem 0;
        text-align: center;
    }

    .metric-label {
        color: #a0aec0 !important;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #68d391 !important;
        font-size: 1.5rem;
        font-weight: bold;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #a0aec0 !important;
        padding: 2rem 0;
        border-top: 1px solid #4a5568;
        margin-top: 2rem;
    }

    /* Fix expander content */
    .streamlit-expanderContent {
        background-color: #1a202c !important;
        color: #ffffff !important;
    }

    /* Generation method info */
    .generation-info {
        background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%) !important;
        color: #ffffff !important;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-left: 4px solid #b794f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

def load_document(uploaded_file):
    """Load PDF document into the QA system."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Initialize QA system and load document
        qa_system = PDFRagQA()
        success = qa_system.load_document(tmp_file_path)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        if success:
            st.session_state.qa_system = qa_system
            st.session_state.document_loaded = True
            st.session_state.document_name = uploaded_file.name
            st.session_state.qa_history = []
            return True
        else:
            return False

    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return False

def ask_question(question):
    """Ask question and get answer from the QA system."""
    if st.session_state.qa_system is None:
        return None

    try:
        result = st.session_state.qa_system.ask_question(question)
        return result
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None

def display_metrics(doc_info):
    """Display document metrics in an appealing way."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📄 Pages</div>
            <div class="metric-value">{doc_info.get('pages', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">✂️ Segments</div>
            <div class="metric-value">{doc_info.get('segments', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📊 Status</div>
            <div class="metric-value">✅ Ready</div>
        </div>
        """, unsafe_allow_html=True)

def format_answer_text(text):
    """Format answer text for better display."""
    if not text:
        return ""

    # Handle different types of formatting
    text = str(text)

    # Convert newlines to proper breaks
    text = text.replace('\n\n', '\n').replace('\n', '\n\n')

    # Handle bullet points
    text = text.replace('• ', '\n• ')
    text = text.replace('- ', '\n• ')

    # Handle numbered lists
    import re
    text = re.sub(r'(\d+\.)', r'\n\1', text)

    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# Main UI
def main():
    # Header with improved dark mode visibility
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PDF RAG QA System</h1>
        <p>Upload a PDF document and ask questions about its content</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for document info and settings
    with st.sidebar:
        st.markdown("### 📊 Document Information")

        if st.session_state.document_loaded:
            doc_info = st.session_state.qa_system.get_document_info()
            st.success(f"📄 **Loaded:** {st.session_state.document_name}")

            # Display metrics
            display_metrics(doc_info)

            if st.button("🗑️ Clear Document", type="secondary"):
                st.session_state.qa_system = None
                st.session_state.document_loaded = False
                st.session_state.document_name = ""
                st.session_state.qa_history = []
                st.rerun()
        else:
            st.warning("No document loaded")

        st.markdown("### ⚙️ System Settings")
        st.info("📌 **Model:** sentence-transformers")
        st.info("🔍 **Search:** FAISS semantic search")
        st.info("🧠 **Generation:** Enhanced RAG pipeline")

        # Show generation capabilities if available
        if st.session_state.document_loaded and hasattr(st.session_state.qa_system, 'get_document_info'):
            doc_info = st.session_state.qa_system.get_document_info()
            if 'generation_capabilities' in doc_info:
                gen_caps = doc_info['generation_capabilities']
                st.markdown("### 🧠 AI Capabilities")
                st.info(f"🌐 OpenAI: {'✅' if gen_caps.get('openai_available') else '❌'}")
                st.info(f"🏠 Local LLM: {'✅' if gen_caps.get('local_llm_loaded') else '❌'}")
                st.info(f"⚙️ Mode: {gen_caps.get('generation_mode', 'basic')}")

        # Theme toggle info
        st.markdown("### 🌙 Dark Mode")
        st.success("✅ Dark mode optimized!")
    # Main content area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 Upload PDF Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze. Maximum size: 200MB"
        )

        if uploaded_file is not None:
            if not st.session_state.document_loaded or st.session_state.document_name != uploaded_file.name:
                with st.spinner("📖 Processing PDF document..."):
                    if load_document(uploaded_file):
                        st.success(f"✅ Document loaded successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to load document. Please try another PDF.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="qa-section">', unsafe_allow_html=True)
        st.subheader("💬 Ask Questions")

        if st.session_state.document_loaded:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                help="Ask any question about the uploaded PDF content"
            )

            col_ask, col_clear = st.columns([3, 1])
            with col_ask:
                ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
            with col_clear:
                if st.button("🧹 Clear History", type="secondary"):
                    st.session_state.qa_history = []
                    st.rerun()

            # Process question
            if ask_button and question.strip():
                with st.spinner("🤖 AI is thinking..."):
                    result = ask_question(question.strip())

                    if result:
                        # Add to history
                        st.session_state.qa_history.append({
                            'question': question.strip(),
                            'result': result,
                            'timestamp': time.strftime("%H:%M:%S")
                        })
                        st.rerun()

            # Display Q&A history
            if st.session_state.qa_history:
                st.subheader("📝 Q&A History")

                # Reverse order to show latest first
                for i, qa in enumerate(reversed(st.session_state.qa_history)):
                    with st.expander(f"❓ {qa['question'][:60]}{'...' if len(qa['question']) > 60 else ''} ({qa['timestamp']})", expanded=(i==0)):

                        # FIXED: Display answer with proper formatting
                        formatted_answer = format_answer_text(qa["result"]["answer"])

                        st.markdown(f"""
                        <div class="answer-box">
                            <strong>🤖 Answer:</strong><br><br>
                            {formatted_answer}
                        </div>
                        """, unsafe_allow_html=True)

                        # Show generation method if available
                        if 'generation_method' in qa["result"]:
                            st.markdown(f"""
                            <div class="generation-info">
                                🧠 Generated using: <strong>{qa["result"]["generation_method"]}</strong> method
                            </div>
                            """, unsafe_allow_html=True)

                        # Display confidence with better visibility
                        confidence = qa["result"].get("confidence", 0) * 100
                        if confidence >= 80:
                            conf_class = "confidence-high"
                            conf_icon = "🟢"
                        elif confidence >= 60:
                            conf_class = "confidence-medium"
                            conf_icon = "🟡"
                        else:
                            conf_class = "confidence-low"
                            conf_icon = "🔴"

                        st.markdown(f'<p class="{conf_class}">{conf_icon} Confidence: {confidence:.1f}%</p>', unsafe_allow_html=True)

                        # FIXED: Display sources with enhanced styling
                        if qa["result"].get("sources"):
                            st.markdown("**📚 Sources:**")
                            for j, source in enumerate(qa["result"]["sources"], 1):
                                similarity = source.get("similarity", 0) * 100
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>📄 Page {source["page"]}</strong> (Relevance: {similarity:.1f}%)
                                    <small>{source["text_preview"]}</small>
                                </div>
                                """, unsafe_allow_html=True)

        else:
            st.info("👆 Please upload a PDF document first to start asking questions.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced footer
    st.markdown("""
    <div class="footer">
        <p>🚀 <strong>PDF RAG QA System</strong> | Powered by sentence-transformers, FAISS, and Streamlit</p>
        <p>🌙 Dark Mode Optimized | 🔒 Local Processing | 📊 Real-time Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
