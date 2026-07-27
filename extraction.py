import time
import json
import os
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from config import parser, format_instructions, HF_TOKEN
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0},
    token=HF_TOKEN
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    max_length=None,
    temperature=0.3,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
chat_model = ChatHuggingFace(llm=llm)

extraction_prompt = ChatPromptTemplate.from_template(
    "Extract structured info from this CTF writeup.\n{format_instructions}\n\nWriteup:\n{writeup}"
)
extraction_chain = extraction_prompt | chat_model | parser

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


def extract_writeup(raw_text, source_path):
    """Returns (Document, raw_result_dict) or (None, None) on failure."""
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


def extract_all(docs):
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
        clean, result = extract_writeup(doc.page_content, source)

        if clean:
            clean_docs.append(clean)
            new_docs.append(clean)
            append_progress(source, result)
        else:
            failed.append(source)

        time.sleep(1)

    print(f"\nDone: {len(clean_docs)} total ({len(new_docs)} new), {len(failed)} failed")
    return clean_docs, new_docs, failed