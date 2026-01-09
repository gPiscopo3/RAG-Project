import streamlit as st
import os
import chromadb
import time
import json
import logging
from utils.pdf_utils import process_pdf_to_chroma_db
from common.rag_response import generate_rag_response
from common.rag_response import highlight_relevant_passages
from utils.chromadb_utils import delete_chroma_collection
from common import config

logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def clear_history():
    st.session_state.messages = []

def download_chat():
    if "messages" not in st.session_state or not st.session_state.messages:
        st.warning("No chat history to download.")
        return
    
    chat_history = st.session_state.messages
    chat_json = json.dumps(chat_history, indent=2)
    return chat_json

st.title("RAG System with Local LLM and ChromaDB")
st.text("Upload a PDF document then ask questions about its content.")

chromadb_client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
list_collections = chromadb_client.list_collections()

# Side menu
with st.sidebar:
    
    st.header("ChromaDB Collections", divider=True)

    st.subheader("Existing Collections")
    if list_collections:
        collection_names = [collection.name for collection in list_collections]
        selected_collection = st.selectbox(
            "Select a collection to query:",
            options=collection_names,
        )

    else:
        st.write("No collections found.")

    # Refresh button to reload the collections
    if st.button(label="Refresh Collections", type="secondary", use_container_width=False, icon=":material/refresh:"):
        list_collections = chromadb_client.list_collections()
        st.rerun()

    # Remove collection Button
    st.header("Remove Collection", divider="red")
    if list_collections:
        collection_names = [collection.name for collection in list_collections]
        collection_to_remove = st.selectbox(
            "Select a collection to remove:",
            options=collection_names,
            key="remove_collection_selectbox"
        )
        if st.button(label="Remove Collection", type="primary", use_container_width=False, icon=":material/delete:", key="initiate_remove"):
            st.session_state.confirm_delete = True

        if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
            st.warning(f"Are you sure you want to remove '{collection_to_remove}'? This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", type="primary", key="confirm_delete_button", use_container_width=True):
                    try:
                        delete_chroma_collection(collection_name=collection_to_remove)
                        st.success(f"Collection '{collection_to_remove}' has been removed.")
                        st.session_state.confirm_delete = False
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove collection: {e}")
                        st.session_state.confirm_delete = False
                with col2:
                    if st.button("Cancel", type="secondary", key="cancel_delete_button", use_container_width=True):
                        st.session_state.confirm_delete = False
                        st.rerun()
    
    # File uploader for new PDFs
    st.header("Upload New PDF", divider="green")

    uploaded_files = st.file_uploader("Upload File", accept_multiple_files=False, type=["pdf"])

    if uploaded_files is not None:
        with st.spinner(f"Processing {uploaded_files.name}..."):
            original_name = uploaded_files.name
            collection_name = original_name.replace(".pdf", "_collection")
            temp_file_path = f"./temp/temp_{original_name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_files.getbuffer())
            
            process_pdf_to_chroma_db(
                pdf_path=temp_file_path,
                persist_directory=config.PERSIST_DIRECTORY,
                model=config.EMBEDDING_MODEL,
                collection_name=collection_name
            )
            os.remove(temp_file_path)
            
            st.success(f"Processed {original_name} and updated ChromaDB.")

    st.header("Utility Buttons", divider=True)

    st.button("Clear Chat History", type="secondary", use_container_width=True, on_click=clear_history)

    # Download chat history button
    if "messages" in st.session_state and st.session_state.messages:
        st.download_button(
            label="Download Chat History",
            data=download_chat(),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True, 
            icon=":material/download:",
            type="secondary"
        )

# Main area for asking questions
if not list_collections:
    st.warning("Please upload a PDF document to create a ChromaDB collection before asking questions.")
    st.stop()

st.markdown(f"You are querying the collection: **{selected_collection}**")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display message content
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            # Display sources in an expander
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(message["sources"]):
                    if isinstance(source, dict):
                        content = source.get("content", "")
                        metadata = source.get("metadata", {})
                    else:
                        content = source
                        metadata = {}
                    # Visual only relevant sources
                    if ":orange-background[" in content:
                        st.info(f"Source {i+1}:\n{content}")
                        if metadata:
                            page = metadata.get("page") or metadata.get("page_number")
                            keywords = metadata.get("keywords", "")
                            md = ""
                            if page is not None:
                                md += f"- **Page:** {page}\n"
                            if keywords:
                                md += f"- **Keywords:** {keywords}\n"
                            if md:
                                st.markdown(f"<span style='font-size: 0.85em'><b>Other Informations:</b><br>{md}</span>", unsafe_allow_html=True)


# Accept user input
if question := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Generate and add assistant response to chat history
    with st.spinner("Generating RAG response..."):

        chat_history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.messages[: -1]
        ]

        answer, sources = generate_rag_response(
            question=question,
            collection_name=selected_collection,
            embedding_model=config.EMBEDDING_MODEL,
            chat_history=chat_history
        )

        answer, highlighted_sources = highlight_relevant_passages(answer, sources)
        logging.info("Generated response with highlighted sources.")
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": highlighted_sources})
    
    # Rerun the app to display the new messages from history
    st.rerun()