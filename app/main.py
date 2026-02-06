import streamlit as st
import os
import sys
import ast  # For parsing stringified lists in metadata

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import HealthcareRAG
from src.utils import load_documents
from dotenv import load_dotenv

MODEL_NAME = 'gemini-2.5-flash'
# Load environment variables
load_dotenv()

st.set_page_config(page_title="Healthcare RAG Agent", page_icon="🏥", layout="wide")

st.title("🏥 Healthcare RAG Q&A Agent")
st.markdown("""
**Disclaimer:** This tool is for informational purposes only. 
It does not provide medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare professional.
""")

# Sidebar for Model Selection
with st.sidebar:
    st.header("Configuration")
    db_choice = st.radio(
        "Select Knowledge Base:",
        ("Custom Chunking (w/ Images)", "Normal Chunking")
    )
    
    if db_choice == "Custom Chunking (w/ Images)":
        persist_dir = "./data/chroma_db_custom"
    else:
        persist_dir = "./data/chroma_db_normal"

    st.divider()
    st.markdown("### Settings")
    st.info("Ensure GOOGLE_API_KEY is set in .env")

# Initialize RAG Agent based on selection
if "rag_agent" not in st.session_state or st.session_state.get("current_db") != persist_dir:
    if os.path.exists(persist_dir):
        st.session_state.rag_agent = HealthcareRAG(persist_directory=persist_dir, model_name=MODEL_NAME)
        st.session_state.current_db = persist_dir
        # st.success(f"Loaded Knowledge Base: {db_choice}")
    else:
        st.error(f"Database not found at {persist_dir}. Please run `python ingest_data.py` first.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path), width=400)

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
                
                # Format text response
                full_response = f"{answer_text}\n\n**Sources:**"
                seen_sources = set()
                images_to_display = []
                
                for doc in source_docs:
                    # Handle both dict and object access
                    if isinstance(doc, dict):
                        meta = doc.get("metadata", {})
                    else:
                        meta = doc.metadata

                    source_name = os.path.basename(meta.get('source', 'Unknown'))
                    page = meta.get('page', 'N/A')
                    source_key = f"{source_name} (Page {page})"
                    
                    if source_key not in seen_sources:
                        full_response += f"\n- {source_key}"
                        seen_sources.add(source_key)

                st.markdown(full_response)

                with st.expander("Show Retrieved Context Details"):
                    for idx, doc in enumerate(source_docs):
                         # Handle both dict and object access
                        if isinstance(doc, dict):
                            meta = doc.get("metadata", {})
                            content = doc.get("page_content", "")
                        else:
                            meta = doc.metadata
                            content = doc.page_content
                        
                        source_name = os.path.basename(meta.get('source', 'Unknown'))
                        page = meta.get('page', 'N/A')
                        st.markdown(f"**{idx+1}. {source_name} (Page {page})**")
                        st.text(content[:300] + "..." if len(content) > 300 else content)
                        st.divider()

                # Display Images
                if images_to_display:
                    st.markdown("### Relevant Images:")
                    cols = st.columns(min(len(images_to_display), 3)) 
                    for idx, img_path in enumerate(images_to_display):
                        if os.path.exists(img_path):
                            # Cycle through columns
                            with cols[idx % 3]:
                                st.image(img_path, caption=f"From {os.path.basename(img_path)}", use_container_width=True)

                # Save to history
                message_data = {"role": "assistant", "content": full_response}
                if images_to_display:
                    message_data["images"] = images_to_display
                
                st.session_state.messages.append(message_data)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")


