# Local-RAG: Motore di Ricerca Semantico

## 📌 Descrizione del Progetto

Questo progetto implementa un sistema di **Retrieval-Augmented Generation (RAG)** completamente locale. L'obiettivo è permettere agli utenti di interrogare i propri documenti (PDF) utilizzando modelli linguistici di grandi dimensioni (LLM), garantendo la privacy dei dati e l'assenza di costi di API esterne.

A differenza di una semplice chat, questo sistema utilizza un approccio di **ingegneria dei dati** per indicizzare i contenuti in un database vettoriale, permettendo ricerche semantiche ad alta precisione.

## 🏗️ Architettura del Sistema

Il sistema è diviso in tre pipeline principali:

1. **Data Ingestion Pipeline**: Caricamento dei documenti, pulizia del testo e suddivisione in chunk (Recursive Character Splitting).
2. **Embedding & Vector Storage**: Trasformazione dei chunk in vettori densi tramite il modello `mxbai-embed-large` e memorizzazione in **ChromaDB**.
3. **RAG Cycle**:
* Ricezione della query.
* Ricerca per similarità (Cosine Similarity) nel DB vettoriale.
* Costruzione del prompt arricchito (Context Injection).
* Generazione della risposta tramite **Llama 3** (via Ollama).

## 🛠️ Stack Tecnologico

* **Orchestration**: [LangChain](https://python.langchain.com/)
* **LLM & Embeddings**: [Ollama](https://ollama.com/) (Llama 3, nomic-embed-text)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
