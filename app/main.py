import streamlit as st
import os
import sys

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import HealthcareRAG
from src.utils import load_documents
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Healthcare RAG Agent", page_icon="🏥")

st.title("🏥 Healthcare RAG Q&A Agent")
st.markdown("""
**Disclaimer:** This tool is for informational purposes only. 
It does not provide medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare professional.
""")

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = HealthcareRAG()

# Check if vector store exists, if not, prompt to ingest
if not os.path.exists("./data/chroma_db"):
    st.warning("⚠️ Knowledge Base not initialized. Found documents in `data/documents/`. initializing now...")
    with st.spinner("Building Knowledge Base for the first time... (This may take a minute)"):
        docs = load_documents("./data/documents")
        if docs:
            st.session_state.rag_agent.ingest_documents(docs)
            st.success(f"✅ Ready! Ingested {len(docs)} pages.")
            st.rerun()
        else:
            st.error("No PDFs found in `data/documents/`. Please add files and restart.")

with st.sidebar:
    st.header("Admin / Setup")
    if st.button("Reload Knowledge Base"):
        with st.spinner("Loading and ingesting documents..."):
            docs = load_documents("./data/documents")
            if docs:
                st.session_state.rag_agent.ingest_documents(docs)
                st.success(f"Ingested {len(docs)} document pages!")
            else:
                st.warning("No PDF documents found in data/documents/")

    st.divider()
    st.markdown("### Settings")
    st.info("Ensure GOOGLE_API_KEY is set in .env")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a healthcare question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Consulting knowledge base..."):
            try:
                response = st.session_state.rag_agent.get_answer(prompt)
                answer_text = response['result']
                source_docs = response['source_documents']
                
                # Format response
                full_response = f"{answer_text}\n\n**Sources:**"
                seen_sources = set()
                for doc in source_docs:
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page = doc.metadata.get('page', 'N/A')
                    source_key = f"{source_name} (Page {page})"
                    if source_key not in seen_sources:
                        full_response += f"\n- {source_key}"
                        seen_sources.add(source_key)
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

