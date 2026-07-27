from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import HF_TOKEN, EMBEDDING_MODEL_REPO

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_REPO)

def build_and_save(clean_docs, path="ctf_corpus"):
    store = FAISS.from_documents(clean_docs, embeddings)
    store.save_local(path)
    return store

def load(path="ctf_corpus"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)