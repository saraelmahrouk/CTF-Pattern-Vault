import time
import json
import os
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from config import parser, format_instructions, HF_TOKEN
from transformers import pipeline
from model_loader import get_chat_model
import torch



extraction_prompt = ChatPromptTemplate.from_template(
    "Extract structured info from this CTF writeup.\n{format_instructions}\n\nWriteup:\n{writeup}"
    "Always return JSON objects"
)

PROGRESS_FILE = "extraction_progress.jsonl"


def _build_document(result, source_path):
    tools = result.get('tools_used', [])
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",")]

    normalized_text = (
        f"Vulnerability: {result.get('vulnerability_class', 'unknown')}\n"
        f"Tools: {', '.join(tools)}\n"
        f"Difficulty: {result.get('difficulty', 'unknown')}\n"
        f"Key insight: {result.get('key_insight', '')}"
    )
    return Document(
        page_content=normalized_text,
        metadata={
            "source": source_path,
            "vulnerability_class": result.get("vulnerability_class"),
            "difficulty": result.get("difficulty"),
        },
    )


def build_extraction_chain(chat_model):
    return extraction_prompt | chat_model | parser

def extract_writeup(raw_text, source_path, chat_model):
    extraction_chain = build_extraction_chain(chat_model)
    try:
        result = extraction_chain.invoke({"format_instructions": format_instructions, "writeup": raw_text})
    except Exception as e:
        print(f"Failed to extract {source_path}: {e}")
        return None, None

    doc = _build_document(result, source_path)
    return doc, result


def load_progress():
    """Returns dict mapping source path -> raw result dict, for already-extracted writeups."""
    if not os.path.exists(PROGRESS_FILE):
        return {}
    done = {}
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            done[entry["source"]] = entry["result"]
    return done


def append_progress(source, result):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": source, "result": result}) + "\n")


def extract_all(docs, chat_model):
    """
    Returns (all_clean_docs, newly_extracted_docs, failed).
    all_clean_docs: every successfully extracted Document (skipped + new) — use for full rebuilds.
    newly_extracted_docs: only the ones extracted THIS run — use for incremental add_documents().
    """
    already_done = load_progress()
    clean_docs, new_docs, failed = [], [], []

    for i, doc in enumerate(docs):
        source = doc.metadata["source"]

        if source in already_done:
            print(f"Skipping {i+1}/{len(docs)}: {source} (already extracted)")
            clean = _build_document(already_done[source], source)
            clean_docs.append(clean)
            continue

        print(f"Processing {i+1}/{len(docs)}: {source}")
        clean, result = extract_writeup(doc.page_content, source, chat_model)

        if clean:
            clean_docs.append(clean)
            new_docs.append(clean)
            append_progress(source, result)
        else:
            failed.append(source)

        time.sleep(15)

    print(f"\nDone: {len(clean_docs)} total ({len(new_docs)} new), {len(failed)} failed")
    return clean_docs, new_docs, failed


def build_training_data(progress_file="extraction_progress.jsonl", output_file="training_data.jsonl"):
    with open(progress_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            entry = json.loads(line)
            result = entry["result"]

            # skip junk/empty extractions (the "None" ones you found earlier)
            if result.get("vulnerability_class") in (None, "None"):
                continue

            example = {
                "prompt": f"Extract structured info from this CTF writeup.\n\nWriteup:\n{entry['source']}",
                "completion": json.dumps(result)
            }
            out.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    build_training_data()