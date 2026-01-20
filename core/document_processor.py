import chromadb
import logging
import config
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.file_helper import extract_tables_from_pdf, extract_images_from_pdf, normalize_text  

logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def delete_chroma_collection(
                              collection_name: str, 
                              persist_directory: str = config.PERSIST_DIRECTORY
                            ):
    """
    Deletes a specified collection from the Chroma database.

    Args:
        collection_name (str): Name of the collection to delete.
        persist_directory (str): Directory where the Chroma database is stored.
    """
    if collection_name is None or collection_name.strip() == "":
        raise ValueError("A valid collection name must be provided.")

    chroma_client = chromadb.PersistentClient(path=persist_directory)

    try:
        chroma_client.delete_collection(name=collection_name)
        logging.info("Collection '%s' has been deleted from ChromaDB.", collection_name)
    except chromadb.errors.NotFoundError:
        logging.warning("Collection '%s' does not exist in ChromaDB.", collection_name)

def process_pdf_to_chroma_db(
    pdf_path            = None,
    chunk_size          = config.CHUNK_SIZE,
    chunk_overlap       = config.CHUNK_OVERLAP,
    model               = config.EMBEDDING_MODEL,
    persist_directory   = config.PERSIST_DIRECTORY,
    collection_name     = None
):
    
    """
    Processes a PDF file, extracts text, tables, and image captions,
    splits the content into chunks, and creates a Chroma database.
    """ 

    if pdf_path is None or pdf_path.strip() == "":
        raise ValueError("A valid PDF path must be provided.")

    if collection_name is None:
        collection_name = pdf_path.split("/")[-1].replace(".pdf", "_collection")

    chroma_client = chromadb.PersistentClient(path=persist_directory)
    try:
        if chroma_client.get_collection(name=collection_name):
            logging.info(f"Collection '%s' already exists in ChromaDB. Skipping processing.", collection_name)
            return
    except Exception:
        pass  # Collection does not exist, proceed with processing

    # Extract non-text elements (tables and images)

    table_docs = extract_tables_from_pdf(pdf_path)
    image_docs = extract_images_from_pdf(pdf_path)

    # Extract text using LangChain's PyMuPDFLoader
    logging.info("Extracting text with PyMuPDFLoader...")
    loader = PyMuPDFLoader(pdf_path)
    text_pages = loader.load()
    logging.info(f"Extracted {len(text_pages)} pages of text.")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        
        separators=[
            # --- For Code and Markdown ---
            "\n```\n",  # Code blocks
            "\n## ",     # Markdown H2 headers
            "\n### ",    # Markdown H3 headers
            "\n#### ",   # Markdown H4 headers
            # --- For Code ---
            "\nclass ",
            "\ndef ",
            "\n\tdef ",
            "\npublic ",        # Java/C#/Kotlin class or method
            "\nprivate ",       # Java/C#/Kotlin method
            "\nprotected ",     # Java/C#/Kotlin method
            "\nfunction ",      # JavaScript/TypeScript function
            "\nfunc ",          # Go/Swift function
            "\npackage ",       # Java/Go package
            "\nimport ",        # General import statement
            "\nmodule ",        # Ruby/Elixir module
            "\nBEGIN ",         # Perl block
            "\nsub ",           # Perl subroutine
            "\nvar ",           # JavaScript/Go variable
            "\nlet ",           # JavaScript/TypeScript variable
            "\nconst ",         # JavaScript/TypeScript constant
            "\nSELECT ",        # SQL select statement
            "\nCREATE ",        # SQL create statement
            "\nINSERT ",        # SQL insert statement
            "\nUPDATE ",        # SQL update statement
            "\nDELETE ",        # SQL delete statement
            # --- For Structured Text and Paragraphs ---
            "\n\n",      # Double newline (paragraphs)
            "\n",        # Newline
            # --- For Sentences and Words ---
            ". ",        # Periods followed by a space
            " ",         # Spaces
            ""           # Characters (fallback)
        ],
        keep_separator=False
    )
    text_docs = text_splitter.split_documents(text_pages)

    # Normalize the content of text chunks
    for doc in text_docs:
        doc.page_content = normalize_text(doc.page_content)

    logging.info(f"Splitted text into {len(text_docs)} chunks.")

    # Combine all documents
    docs = text_docs + table_docs + image_docs
    logging.info(f"Total documents to be indexed: {len(docs)}")
    logging.info(f"Text documents: {len(text_docs)}, Table documents: {len(table_docs)}, Image documents: {len(image_docs)}")

    # Initialize embeddings and create the Chroma database
    embeddings = OllamaEmbeddings(model=model)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    logging.info("Chroma database created at %s with collection name '%s'.", persist_directory, collection_name)
