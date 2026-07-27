from vectorstore import load

vs = load()

# FAISS keeps documents in an internal docstore dict
docstore = vs.docstore._dict

print(f"Total documents in corpus: {len(docstore)}")

for doc_id, doc in docstore.items():
    print(doc_id)
    print(doc.page_content)
    print(doc.metadata)
    print("---")