from common.rag_response import generate_rag_response
from utils.pdf_utils import process_pdf_to_chroma_db

if __name__ == "__main__":
    # Example usage: Process a PDF and create a Chroma database
    pdf_path = "aws-overview.pdf"  # Replace with your PDF file path
    process_pdf_to_chroma_db(pdf_path)

    # Example usage: Generate a RAG response
    question = "What is the main topic of the document?"  # Replace with your question
    answer = generate_rag_response(collection_name='aws-overview_collection', question=question)

    print("RAG Response:", answer)
    