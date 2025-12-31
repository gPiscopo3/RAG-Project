import chromadb
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def delete_chroma_collection(
                              collection_name: str, 
                              persist_directory: str = "./chroma_db"
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
