import os
from loaders import clone_repos, load_writeups
from extraction import extract_all
from vectorstore import build_and_save, load
from config import CORPUS_PATH
from model_loader import get_chat_model

INDEX_FILE = os.path.join(CORPUS_PATH, "index.faiss")

def main():
    chat_model = get_chat_model(use_local=False)
    index_exists = os.path.exists(INDEX_FILE)

    clone_repos(["https://github.com/rsa-ctf/write-ups"])
    docs = load_writeups()
    print(f"Loaded {len(docs)} writeup files")

    clean_docs, new_docs, failed = extract_all(docs)
    print(f"Extraction done: {len(clean_docs)} total ({len(new_docs)} new), {len(failed)} failed")

    if not clean_docs:
        print("No documents were successfully extracted — aborting build.")
        return

    if index_exists:
        if not new_docs:
            print("No new writeups to add — corpus unchanged.")
            return
        vs = load(CORPUS_PATH)
        vs.add_documents(new_docs)
        vs.save_local(CORPUS_PATH)
        print(f"Appended {len(new_docs)} new cards to existing corpus at {CORPUS_PATH}")
    else:
        build_and_save(clean_docs, CORPUS_PATH)
        print(f"Corpus built and saved to {CORPUS_PATH}")

if __name__ == "__main__":
    main()