from common.rag_response import generate_rag_response
from utils.pdf_utils import process_pdf_to_chroma_db

if __name__ == "__main__":
    # Example usage: Process a PDF and create a Chroma database
    pdf_path = "aws-overview.pdf"  # Replace with your PDF file path
    process_pdf_to_chroma_db(pdf_path)

    collection_name = 'aws-overview_collection'
    print("Now you can ask questions about the PDF. Type '.exit' to quit.")
    while True:
        question = input("")
        if question.lower() == '.exit':
            break
        answer = generate_rag_response(collection_name=collection_name, question=question)
        print("RAG Response:", answer)

    # Example usage: Delete a Chroma collection 
    # delete_chroma_collection(collection_name='aws-overview_collection')