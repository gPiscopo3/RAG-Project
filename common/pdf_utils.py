from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

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

def process_pdf_to_chroma_db(pdf_path, chunk_size=2000, chunk_overlap=100, persist_directory="./chroma_db", collection_name="local_rag_db"):
    """
    Processa un file PDF, estrae il testo, lo suddivide in chunk, e crea un database Chroma.

    Args:
        pdf_path (str): Percorso del file PDF.
        chunk_size (int): Dimensione di ogni chunk di testo.
        chunk_overlap (int): Sovrapposizione tra i chunk.
        persist_directory (str): Directory in cui salvare il database Chroma.
        collection_name (str): Nome della collezione nel database Chroma.
    """
    # Estrai il testo dal PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)

    # Suddividi il testo in chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(extracted_text)
    print(f"\nSplitted into {len(chunks)} chunks.")

    # Converti i chunk in oggetti Document
    docs = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]

    # Inizializza l'oggetto embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Crea il database Chroma
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"Chroma database created at {persist_directory} with collection name '{collection_name}'.")