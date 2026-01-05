import chromadb
import logging
import camelot
import fitz
from common import config
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def normalize_text(text: str) -> str:
    """Pulisce il testo per l'embedding."""
    return text.replace("\n", " ").replace("\r", " ").strip().lower()

def extract_images_from_pdf (file_path):
    """Estrae immagini e cerca di associare una didascalia testuale vicina."""
    doc = fitz.open(file_path)
    image_docs = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        # Ordina le immagini per posizione verticale per processarle dall'alto verso il basso
        image_list.sort(key=lambda img: page.get_image_bbox(img).y1)

        for img_index, img in enumerate(image_list):
            # Ottieni il bounding box dell'immagine
            img_bbox = page.get_image_bbox(img)

            # Cerca il testo sotto l'immagine (potenziale didascalia)
            # Definiamo un'area di ricerca sotto l'immagine
            search_area = fitz.Rect(img_bbox.x0, img_bbox.y1, img_bbox.x1, img_bbox.y1 + 50)
            text_in_area = page.get_text("text", clip=search_area, sort=True).strip()

            caption = text_in_area if text_in_area else "No caption found"

            # Crea un documento con la descrizione dell'immagine
            content = f"[Image: An image is present on the page. Caption: '{caption}']"
            normalized_content = normalize_text(content)
            metadata = {
                "content_type": "image_caption",
                "page_number": page_num + 1,
                "image_index_on_page": img_index,
                "image_bbox": str([img_bbox.x0, img_bbox.y0, img_bbox.x1, img_bbox.y1])
            }
            image_docs.append(Document(page_content=normalized_content, metadata=metadata))

    logging.info(f"Extracted {len(image_docs)} image captions from {file_path}")
    return image_docs

def extract_tables_from_pdf(file_path):
    """Estrae le tabelle da un PDF e le converte in formato Markdown."""
    try:
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice', suppress_stdout=True)
        table_docs = []
        for i, table in enumerate(tables):
            # Converte il DataFrame della tabella in una stringa Markdown
            markdown_table = tabulate(table.df, headers='keys', tablefmt='pipe')
            normalized_table = normalize_text(markdown_table)
            
            # Crea un metadato per la tabella
            metadata = {
                "content_type": "table",
                "page_number": table.page,
                "table_index_on_page": i
            }
            
            # Crea un Documento LangChain per ogni tabella
            table_docs.append(Document(page_content=normalized_table, metadata=metadata))
        
        logging.info(f"Extracted {len(table_docs)} tables from {file_path}")
        return table_docs
    except Exception as e:
        logging.error(f"Could not extract tables from {file_path}: {e}")
        return []

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

    # 1. Estrai elementi non testuali (tabelle e immagini)
    # table_docs = extract_tables_from_pdf(pdf_path)
    # image_docs = extract_images_from_pdf(pdf_path)

    # 2. Estrai il testo usando PyMuPDFLoader di LangChain
    logging.info("Extracting text with PyMuPDFLoader...")
    loader = PyMuPDFLoader(pdf_path)
    text_pages = loader.load()
    logging.info(f"Extracted {len(text_pages)} pages of text.")

    # 3. Suddividi il testo in chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        
        separators=[
            # --- Per Codice e Markdown ---
            "\n```\n",  # Blocchi di codice
            "\n## ",     # Intestazioni Markdown H2
            "\n### ",    # Intestazioni Markdown H3
            "\n#### ",   # Intestazioni Markdown H4
            # --- Per Codice (ispirato a Python) ---
            "\nclass ",
            "\ndef ",
            "\n\tdef ",
            # --- Per Testo Strutturato e Paragrafi ---
            "\n\n",      # Doppia riga nuova (paragrafi)
            "\n",        # Riga nuova
            # --- Per Frasi e Parole ---
            ". ",        # Punti seguiti da spazio
            " ",         # Spazi
            ""           # Caratteri (fallback)
        ],
        keep_separator=False
    )
    text_docs = text_splitter.split_documents(text_pages)

    # 4. Normalizza il contenuto dei chunk di testo
    for doc in text_docs:
        doc.page_content = normalize_text(doc.page_content)

    logging.info(f"Splitted text into {len(text_docs)} chunks.")

    # 5. Combina tutti i documenti
    # docs = text_docs + table_docs + image_docs
    docs = text_docs
    logging.info(f"Total documents to be indexed: {len(docs)}")
    # logging.info(f"Text documents: {len(text_docs)}, Table documents: {len(table_docs)}, Image documents: {len(image_docs)}")

    # 6. Inizializza gli embeddings e crea il database Chroma
    embeddings = OllamaEmbeddings(model=model)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    logging.info("Chroma database created at %s with collection name '%s'.", persist_directory, collection_name)