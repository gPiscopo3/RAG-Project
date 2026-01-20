import chromadb
import logging
import camelot
import fitz
import config
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from tabulate import tabulate

logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def normalize_text(text: str) -> str:
    """Cleans the text for embedding."""
    return text.replace("\n", " ").replace("\r", " ").strip().lower()

def extract_images_from_pdf (file_path):
    """Extracts images and tries to associate a nearby text caption."""
    doc = fitz.open(file_path)
    image_docs = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        # Sort images by vertical position to process them from top to bottom
        image_list.sort(key=lambda img: page.get_image_bbox(img).y1)

        for img_index, img in enumerate(image_list):
            # Get the bounding box of the image
            img_bbox = page.get_image_bbox(img)

            # Search for text below the image (potential caption)
            # We define a search area below the image
            search_area = fitz.Rect(img_bbox.x0, img_bbox.y1, img_bbox.x1, img_bbox.y1 + 50)
            text_in_area = page.get_text("text", clip=search_area, sort=True).strip()

            caption = text_in_area if text_in_area else "No caption found"

            # Create a document with the image description
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
    """Extracts tables from a PDF and converts them to Markdown format."""
    try:
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice', suppress_stdout=True)
        table_docs = []
        for i, table in enumerate(tables):
            # Converts the table's DataFrame into a Markdown string
            markdown_table = tabulate(table.df, headers='keys', tablefmt='pipe')
            normalized_table = normalize_text(markdown_table)
            
            # Create metadata for the table
            metadata = {
                "content_type": "table",
                "page_number": table.page,
                "table_index_on_page": i
            }
            
            # Create a LangChain Document for each table
            table_docs.append(Document(page_content=normalized_table, metadata=metadata))
        
        logging.info(f"Extracted {len(table_docs)} tables from {file_path}")
        return table_docs
    except Exception as e:
        logging.error(f"Could not extract tables from {file_path}: {e}")
        return []
