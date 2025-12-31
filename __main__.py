import streamlit as st
import os
import chromadb
from utils.pdf_utils import process_pdf_to_chroma_db
from common.rag_response import generate_rag_response

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
    
    # Add file uploader in the sidebar
    st.header("Upload New PDF", divider=True)

    uploaded_files = st.file_uploader("Upload File", accept_multiple_files=False, type=["pdf"])

    if uploaded_files is not None:
        with st.spinner(f"Processing {uploaded_files.name}..."):
            # Usa il nome originale per la collection
            original_name = uploaded_files.name
            collection_name = original_name.replace(".pdf", "_collection")
            temp_file_path = f"./temp_{original_name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_files.getbuffer())
            
            # Passa collection_name esplicitamente
            process_pdf_to_chroma_db(
                pdf_path=temp_file_path,
                persist_directory="./chroma_db",
                model="mxbai-embed-large",
                collection_name=collection_name
            )
            os.remove(temp_file_path)
            
            st.success(f"Processed {original_name} and updated ChromaDB.")

# Main area for asking questions
st.markdown(f"You are querying the collection: **{selected_collection}**")
if question := st.chat_input("Type your question here..."):
    with st.spinner("Generating RAG response..."):
        answer = generate_rag_response(
            question=question,
            collection_name=selected_collection
        )
        st.chat_message("user").markdown(question)
        st.chat_message("assistant").markdown(answer)