# ingestion.py
import numpy as np
from extraction import extract_writeup, append_progress
from clustering import update_with_new
from config import CORPUS_PATH
from vectorstore import load

vectorstore = load(CORPUS_PATH)

def is_duplicate(vectorstore, label):
    existing_sources = {doc.metadata.get("source") for doc in vectorstore.docstore._dict.values()}
    return label in existing_sources

def add_user_writeup(vectorstore, raw_text, label):
    if is_duplicate(vectorstore, label):
        return None, None, f"A writeup labeled '{label}' already exists in the corpus."

    clean_doc, result = extract_writeup(raw_text, source_path=label)
    if clean_doc is None:
        return None, None, "Extraction failed — could not process this writeup."

    append_progress(label, result)
    vectorstore.add_documents([clean_doc])
    vectorstore.save_local(CORPUS_PATH)

    new_embedding = vectorstore.embeddings.embed_query(clean_doc.page_content)
    update_with_new(np.array([new_embedding], dtype=np.float32))

    return clean_doc, result, None


