import streamlit as st
import logging
import config
from core.rag_manager import generate_rag_response, highlight_relevant_passages

logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def render_chat_interface(collection_name: str):    
    st.markdown(f"You are querying the collection: **{collection_name}**")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display message content
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                # Display sources in an expander
                with st.expander("Sources", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        if isinstance(source, dict):
                            content = source.get("content", "")
                            metadata = source.get("metadata", {})
                        else:
                            content = source
                            metadata = {}
                        # Visual only relevant sources
                        if ":orange-background[" in content:
                            st.info(f"Source {i+1}:\n{content}")
                            if metadata:
                                page = metadata.get("page") or metadata.get("page_number")
                                keywords = metadata.get("keywords", "")
                                md = ""
                                if page is not None:
                                    md += f"- **Page:** {page}\n"
                                if keywords:
                                    md += f"- **Keywords:** {keywords}\n"
                                if md:
                                    st.markdown(f"<span style='font-size: 0.85em'><b>Other Informations:</b><br>{md}</span>", unsafe_allow_html=True)


    # Accept user input
    if question := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Generate and add assistant response to chat history
        with st.spinner("Generating RAG response..."):

            chat_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.messages[: -1]
            ]

            answer, sources = generate_rag_response(
                question=question,
                collection_name=collection_name,
                embedding_model=config.EMBEDDING_MODEL,
                chat_history=chat_history
            )

            answer, highlighted_sources = highlight_relevant_passages(answer, sources)
            logging.info("Generated response with highlighted sources.")
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": highlighted_sources})
        
        # Rerun the app to display the new messages from history
        st.rerun()