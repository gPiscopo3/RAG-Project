import streamlit as st
from ui.sidebar import render_sidebar
from ui.chat_interface import render_chat_interface

st.title("RAG System")

selected_collection = render_sidebar()

if selected_collection:
    render_chat_interface(collection_name=selected_collection)
else:
    st.warning("Create or select a collection to start querying.")
