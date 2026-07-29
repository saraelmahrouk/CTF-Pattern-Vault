import streamlit as st
import os
from model_loader import get_chat_model
from vectorstore import load
from rag_chain import build_rag_chain, build_hint_chain, build_coaching_chain
from config import CORPUS_PATH

INDEX_FILE = os.path.join(CORPUS_PATH, "index.faiss")

@st.cache_resource(show_spinner=False)
def load_everything(use_local: bool):
    if not os.path.exists(INDEX_FILE):
        st.error(f"No corpus found at {CORPUS_PATH}. Run `python build_corpus.py` first.")
        st.stop()
    chat_model = get_chat_model(use_local=use_local)
    vectorstore = load(CORPUS_PATH)
    rag_chain = build_rag_chain(vectorstore, chat_model)
    hint_chain = build_hint_chain(vectorstore, chat_model)
    coaching_chain = build_coaching_chain(vectorstore, chat_model)
    return vectorstore, rag_chain, hint_chain, coaching_chain

