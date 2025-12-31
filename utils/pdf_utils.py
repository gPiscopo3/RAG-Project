import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_metadata_from_pdf(file_path):
    reader = PdfReader(file_path)
    metadata = reader.metadata
    return metadata

def extract_images_with_caption(file_path):
    # Placeholder function: Implement image extraction logic if needed
    return []

def extract_tables_with_caption(file_path):
    # Placeholder function: Implement table extraction logic if needed
    return []

def process_pdf_to_chroma_db(
                            pdf_path = None, 
                            chunk_size=2000, 
                            chunk_overlap=100, 
                            persist_directory="./chroma_db" 
                            ):
    """
    Processes a PDF file, extracts the text, splits it into chunks, and creates a Chroma database.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
        persist_directory (str): Directory to save the Chroma database.
        collection_name (str): Name of the collection in the Chroma database.
    """ 
    if pdf_path is None or pdf_path.strip() == "":
        raise ValueError("A valid PDF path must be provided.")

    collection_name = pdf_path.split("/")[-1].replace(".pdf", "_collection")

    # Check if the collection already exists in ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_collection(name=collection_name)
    if collection:
        print(f"Collection '{collection_name}' already exists in ChromaDB. Skipping processing.")
        return

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    print("\nExtracted text from PDF.")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(extracted_text)
    print(f"\nSplitted into {len(chunks)} chunks.")

    # Convert chunks into Document objects
    docs = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]

    # Initialize the embeddings object
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create the Chroma database
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"Chroma database created at {persist_directory} with collection name '{collection_name}'.")