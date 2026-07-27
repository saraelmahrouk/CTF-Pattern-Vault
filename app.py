import os
from vectorstore import load
from rag_chain import build_rag_chain, build_hint_chain

CORPUS_PATH = "ctf_corpus"
INDEX_FILE = os.path.join(CORPUS_PATH, "index.faiss")

if not os.path.exists(INDEX_FILE):
    raise SystemExit(
        f"No corpus found at {CORPUS_PATH}. Run `python build_corpus.py` first to build it."
    )

vectorstore = load(CORPUS_PATH)
rag_chain = build_rag_chain(vectorstore)
hint_chain = build_hint_chain(vectorstore)

if __name__ == "__main__":
    query = "I found a locked PDF with weird encryption, what should I try?"
    result = rag_chain.invoke(query)
    result2 = hint_chain.invoke(query)
    results = load().similarity_search_with_score(query, k=2)

    for doc, score in results:
        print(f"score: {score:.4f}")
        print(doc.page_content[:300])  # or doc.metadata if pattern-card fields are stored there
        print("---")

    print(result)
    print(result2)
