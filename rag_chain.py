from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from extraction import chat_model
from config import hint_format_instructions, hint_parser

rag_prompt = ChatPromptTemplate.from_template(
    "You are a CTF coaching assistant. Use the following pattern cards from past writeups "
    "to help answer the user's question about their current challenge.\n\n"
    "Pattern cards:\n{context}\n\n"
    "Question: {question}"
)

hint_prompt = ChatPromptTemplate.from_template(
    "You are a CTF coaching assistant. A student is stuck on a challenge and asked:\n"
    "{question}\n\n"
    "Here are pattern cards from similar past writeups that may help:\n"
    "{context}\n\n"
    "Generate three graduated hints based on these pattern cards:\n"
    "1. A vague category-level nudge\n"
    "2. A more specific tool/technique nudge\n"
    "3. A near-complete walkthrough (stop short of the final flag)\n\n"
    "{format_instructions}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vectorstore, k=2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": lambda x: x}
        | rag_prompt
        | chat_model
        | StrOutputParser()
    )

def build_hint_chain(vectorstore, k=2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return (
        {
            "context": retriever | format_docs,
            "question": lambda x: x,
            "format_instructions": lambda _: hint_format_instructions,
        }
        | hint_prompt
        | chat_model
        | hint_parser
    )