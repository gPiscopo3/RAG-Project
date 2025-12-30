from ollama import Client
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

ollama_host_url = "http://localhost:11434/"
local_model = "llama3"
embedding_model = "nomic-embed-text"
client = Client(host=ollama_host_url)

vector_db = Chroma(
                persist_directory="./chroma_db", 
                embedding_function=OllamaEmbeddings(model=embedding_model, base_url=ollama_host_url), 
                collection_name="local_rag_db"
                )
retriever = vector_db.as_retriever()

question = "What is the main topic of the document? What is the name of the Authors?"
docs = retriever.invoke(question)
print("Documents fetched from database : "+str(len(docs)))

context = "\n\n".join(doc.page_content for doc in docs)

# Create a RAG prompt in below format
formatted_prompt = f"""Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Call ollama chat api to generate the response from provided context
response = client.chat(model='llama3',messages=[{'role': 'user', 'content': formatted_prompt}])
print(response['message']['content'])
