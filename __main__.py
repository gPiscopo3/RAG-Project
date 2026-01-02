import streamlit as st
import os
import chromadb
from utils.pdf_utils import process_pdf_to_chroma_db
from common.rag_response import generate_rag_response
import time

st.title("RAG System with Local LLM and ChromaDB")
st.text("Upload a PDF document then ask questions about its content.")

chromadb_client = chromadb.PersistentClient(path="./chroma_db")
list_collections = chromadb_client.list_collections()

# Side menu for show between existing collections
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
                        chromadb_client.delete_collection(name=collection_to_remove)
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
    
    # Add file uploader in the sidebar
    st.header("Upload New PDF", divider="green")

    uploaded_files = st.file_uploader("Upload File", accept_multiple_files=False, type=["pdf"])

    if uploaded_files is not None:
        with st.spinner(f"Processing {uploaded_files.name}..."):
            original_name = uploaded_files.name
            collection_name = original_name.replace(".pdf", "_collection")
            temp_file_path = f"./temp_{original_name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_files.getbuffer())
            
            process_pdf_to_chroma_db(
                pdf_path=temp_file_path,
                persist_directory="./chroma_db",
                model="nomic-embed-text",
                collection_name=collection_name
            )
            os.remove(temp_file_path)
            
            st.success(f"Processed {original_name} and updated ChromaDB.")

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
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Generating RAG response..."):
            answer = generate_rag_response(
                question=question,
                collection_name=selected_collection
            )
            st.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})