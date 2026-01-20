import streamlit as st
import logging
import config
import json
import chromadb
import time
import os
from core.document_processor import delete_chroma_collection, process_pdf_to_chroma_db


chromadb_client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)

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

def render_sidebar():
    with st.sidebar:
        st.header("ChromaDB Collections", divider=True)

        st.subheader("Existing Collections")
        
        list_collections = chromadb_client.list_collections()
        selected_collection = None

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
                # Replace spaces with underscores for a valid collection name
                collection_name = original_name.replace(" ", "_").replace(".pdf", "_collection")
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
        
        return selected_collection