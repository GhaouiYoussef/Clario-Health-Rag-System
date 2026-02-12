import streamlit as st
import os
import sys
import ast  # For parsing stringified lists in metadata

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import HealthcareRAG
from src.utils import load_documents
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Healthcare RAG Agent", page_icon="🏥", layout="wide")

# Check for API Key
google_api_key = os.getenv("GOOGLE_API_KEY")

st.title("🏥 Healthcare RAG Q&A Agent")
st.markdown("""
**Disclaimer:** This tool is for informational purposes only. 
It does not provide medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare professional.
""")

# Sidebar for Model Selection
with st.sidebar:
    st.header("Configuration")
    
    # API Key Configuration
    if not google_api_key:
        st.error("⚠️ GOOGLE_API_KEY not found.")
        api_key_input = st.text_input("Enter Google API Key:", type="password")
        if api_key_input:
            # Set in environment for current session
            os.environ["GOOGLE_API_KEY"] = api_key_input
            google_api_key = api_key_input
            
            # Create/Update .env file
            env_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), '.env')
            with open(env_path, "a") as f:
                f.write(f"\nGOOGLE_API_KEY={api_key_input}\n")
            st.success("API Key saved to .env!")
            st.rerun()

    model_name = st.selectbox(
        "Select Model:",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0
    )

    db_choice = st.radio(
        "Select Knowledge Base:",
        ("Custom Chunking (w/ Images)", "Normal Chunking")
    )
    
    if db_choice == "Custom Chunking (w/ Images)":
        persist_dir = "./data/chroma_db_test_json"
    else:
        persist_dir = "./data/chroma_db_test_json" # Default for now


    st.divider()
    st.markdown("### Settings")
    n_results = st.slider("Num Results (Final)", 1, 10, 5)
    k_candidates = st.slider("Num Candidates (Initial)", 10, 50, 40)
    doc_div = st.slider("Document Diversity", 0.1, 1.0, 0.5)
    
    if google_api_key:
        st.success("API Key is configured.")
    else:
        st.info("Ensure GOOGLE_API_KEY is set in .env or provide it above.")

# Initialize RAG Agent based on selection
if ("rag_agent" not in st.session_state or 
    st.session_state.get("current_db") != persist_dir or 
    st.session_state.get("current_model") != model_name or
    st.session_state.get("current_api_key") != google_api_key):
    
    if os.path.exists(persist_dir) and google_api_key:
        st.session_state.rag_agent = HealthcareRAG(
            persist_directory=persist_dir, 
            model_name=model_name,
            api_key=google_api_key
        )
        st.session_state.current_db = persist_dir
        st.session_state.current_model = model_name
        st.session_state.current_api_key = google_api_key
        st.success(f"Loaded {db_choice} with {model_name}")
    elif not google_api_key:
        st.warning("Please provide a Google API Key in the sidebar to initialize the agent.")
    else:
        st.error(f"Database not found at {persist_dir}. Please run `python ingest_from_json.py` first.")

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
        message_placeholder = st.empty()
        status_text = st.status("Thinking...", expanded=True)
        
        def update_status(step, msg):
            status_text.write(f"**{step.title()}:** {msg}")
            if step == 'complete':
                status_text.update(label="Response Ready", state="complete", expanded=False)

        try:
            # Use user-configured parameters from sidebar
            response = st.session_state.rag_agent.get_answer(
                prompt,
                n_results=n_results,
                k_candidates=k_candidates,
                doc_diversity=doc_div,
                status_callback=update_status
            )
            answer_text = response['result']
            source_docs = response['source_documents']
            
            # Format text response
            full_response = f"{answer_text}\n\n**Sources:**"
            seen_sources = set()
            images_to_display = []
            
            for doc in source_docs:
                # Handle both dict and object access safely
                if isinstance(doc, dict):
                    meta = doc.get("metadata", {})
                    # Ensure chunk id is available for link
                    chunk_id = doc.get("id", "unknown")
                else:
                    meta = doc.metadata
                    # If using old object, ID might not be easily accessible unless added to object
                    chunk_id = getattr(doc, "id", "unknown")

                # Improved source mapping logic
                raw_source = meta.get('doc_name', meta.get('source', 'Unknown'))
                source_name = os.path.basename(raw_source)
                page = meta.get('page_range', meta.get('page', 'N/A'))
                title = meta.get('title', '')
                
                source_key = f"{source_name} (Page {page})"
                if title:
                    source_key = f"{title} - {source_key}"
                
                # Check for duplicate
                if chunk_id not in seen_sources:
                    full_response += f"\n- {source_key} `[{chunk_id[:8]}...]`"
                    seen_sources.add(chunk_id)

            message_placeholder.markdown(full_response)

            with st.expander("Show Retrieved Context Details"):
                for idx, doc in enumerate(source_docs):
                     # Handle both dict and object access
                    if isinstance(doc, dict):
                        meta = doc.get("metadata", {})
                        content = doc.get("page_content", "")
                        chunk_id = doc.get("id", "Unknown ID")
                        score = doc.get("score", 0.0)
                    else:
                        meta = doc.metadata
                        content = doc.page_content
                        chunk_id = getattr(doc, "id", "Unknown ID")
                        score = getattr(doc, "score", 0.0)
                    
                    raw_source = meta.get('doc_name', meta.get('source', 'Unknown'))
                    source_name = os.path.basename(raw_source)
                    page = meta.get('page_range', meta.get('page', 'N/A'))
                    title = meta.get('title', 'Unknown Section')
                    
                    st.markdown(f"**{idx+1}. {title}** `(Score: {score:.4f})`")
                    st.caption(f"File: {source_name} | Page: {page} | Chunk ID: `{chunk_id}`")
                    
                    # Collapsible full content
                    with st.expander("View Full Chunk Content"):
                        st.text(content)
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
                
            with st.expander(f"Source {idx+1}: {title} ({source_name})"):
                st.caption(f"**Chunk ID:** `{chunk_id}` | **Page:** {page}")
                st.markdown(f"**Context Snippet:**")
                st.info(content) # Show full chunk content
                
                # Image Discovery Logic (Simple heuristic based on page number)
                # If we have page info and doc name, try to find image
                # doc_001.pdf -> images/doc_001/page_X_img_Y.png
                if page != 'N/A' and page != 'unknown':
                    # Try to find images for this page
                    # page might be "5" or "5-6"
                    first_page = page.split('-')[0] if '-' in str(page) else str(page)
                    clean_doc_name = source_name.replace('.pdf', '')
                    img_search_dir = os.path.join("data", "images", clean_doc_name)
                    
                    if os.path.exists(img_search_dir):
                        found_imgs = [
                            os.path.join(img_search_dir, f) 
                            for f in os.listdir(img_search_dir) 
                            if f"page_{first_page}_" in f
                        ]
                        if found_imgs:
                            st.markdown("**Related Images:**")
                            st.image(found_imgs, width=300)
                            images_to_display.extend(found_imgs)

            # Save to history
            message_data = {"role": "assistant", "content": answer_text, "references": source_docs, "images": images_to_display}
            st.session_state.messages.append(message_data)
        
        except Exception as e:
            status_placeholder.update(label="Error", state="error")
            st.error(f"An error occurred: {e}")


