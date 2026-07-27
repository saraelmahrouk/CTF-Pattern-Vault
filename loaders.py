import os
import subprocess
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

def clone_repos(repo_urls, dest_dir="data"):
    for repo in repo_urls:
        subprocess.run(["git", "clone", "--depth", "1", repo, ], cwd=dest_dir)

def load_writeups(source_dir="data"):
    docs = []
    md_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))

    for path in md_files:
        loaded = None
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                loaded = TextLoader(path, encoding=enc).load()
                break
            except Exception:
                continue

        if loaded:
            docs.extend(loaded)
        else:
            print(f"Could not load {path} with any known encoding")

    return docs