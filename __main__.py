# from common.rag_response import generate_rag_response
# from utils.pdf_utils import process_pdf_to_chroma_db

# if __name__ == "__main__":
#     # Example usage: Process a PDF and create a Chroma database
#     pdf_path = "aws-overview.pdf"  # Replace with your PDF file path
#     process_pdf_to_chroma_db(pdf_path)

#     collection_name = 'aws-overview_collection'
#     print("Now you can ask questions about the PDF. Type '.exit' to quit.")
#     while True:
#         question = input("")
#         if question.lower() == '.exit':
#             break
#         answer = generate_rag_response(collection_name=collection_name, question=question)
#         print("RAG Response:", answer)

#     # Example usage: Delete a Chroma collection 
#     # delete_chroma_collection(collection_name='aws-overview_collection')
import streamlit as st
import os
from utils.pdf_utils import process_pdf_to_chroma_db
from common.rag_response import generate_rag_response

st.title("RAG System with Local LLM and ChromaDB")

uploaded_files = st.file_uploader(
    "Upload File", accept_multiple_files=False, type=["pdf"]
)

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

    st.write("You can now ask questions about the uploaded PDF.")
    if question := st.chat_input("Type your question here..."):
        with st.spinner("Generating RAG response..."):
            answer = generate_rag_response(
                question=question,
                collection_name=uploaded_files.name.replace(".pdf", "_collection")
            )
            st.chat_message("user").markdown(question)
            st.chat_message("assistant").markdown(answer)