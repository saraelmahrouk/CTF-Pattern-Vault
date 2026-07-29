from vectorstore import load

vs = load()

# FAISS keeps documents in an internal docstore dict
docstore = vs.docstore._dict

# vs.delete(ids="")
# vs.save_local("ctf_corpus")
# sources_to_remove = [""]
# ids_to_remove = [
#     doc_id for doc_id, doc in vs.docstore._dict.items()
#     if doc.metadata.get("source") in sources_to_remove
# ]
# print(f"Found {len(ids_to_remove)} matching entries to remove")
# vs.delete(ids=ids_to_remove)
# vs.save_local("ctf_corpus")
print(f"Total documents in corpus: {len(docstore)}")

for doc_id, doc in docstore.items():
    print(doc_id)
    print(doc.page_content)
    print(doc.metadata)
    print("---")