import os
from vectorstore import load
from rag_chain import build_rag_chain, build_hint_chain, build_coaching_chain

CORPUS_PATH = "ctf_corpus"
INDEX_FILE = os.path.join(CORPUS_PATH, "index.faiss")

if not os.path.exists(INDEX_FILE):
    raise SystemExit(
        f"No corpus found at {CORPUS_PATH}. Run `python build_corpus.py` first to build it."
    )

vectorstore = load(CORPUS_PATH)
rag_chain = build_rag_chain(vectorstore)
hint_chain = build_hint_chain(vectorstore)
coaching_chain = build_coaching_chain(vectorstore)

if __name__ == "__main__":
    query = "I am noticing weak credentials, what could be the next step?"
    result = rag_chain.invoke(query)
    result2 = hint_chain.invoke(query)
    summary = coaching_chain.invoke("XOR encryption challenges")
    
    print(summary)
    print(result)
    print(result2)
