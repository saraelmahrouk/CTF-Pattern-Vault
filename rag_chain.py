from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from config import hint_format_instructions, hint_parser
from clustering import cluster_biased_retrieve, get_cluster_topic_docs

rag_prompt = ChatPromptTemplate.from_template(
    "You are a CTF coaching assistant. Use the following pattern cards from past writeups "
    "to help answer the user's question about their current challenge.\n\n"
    "If the question is not related to a CTF challenge, or if none of the pattern cards below "
    "are actually relevant to the question, say so plainly instead of guessing or giving a vague, "
    "generic answer.\n\n"
    "If you are not fully certain about a specific technical detail (a formula, a command, "
    "an exact method), say so explicitly rather than stating it confidently. It is better "
    "to describe the general approach than to state something specific that might be incorrect.\n\n"
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

coaching_prompt = ChatPromptTemplate.from_template(
    "You are a CTF coaching assistant. A student wants to learn about the following topic:\n"
    "{question}\n\n"
    "Below are notes from several past challenges related to this topic. Synthesize them into "
    "one clear, concise summary of what you know about this topic overall — key techniques, "
    "common tools, and useful insights. Write it as a natural knowledge summary, not as a list of "
    "separate writeups, and do not mention where this information came from or how it was gathered.\n\n"
    "Keep the summary tight and focused: aim for 4-6 sentences covering the most important patterns, "
    "not an exhaustive account of every detail in the notes. Prioritize the most common or important "
    "techniques over rare edge cases.\n\n"
    "Notes:\n{context}\n\n"
    "Summary:"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_cluster_retriever(vectorstore, k=2):
    return RunnableLambda(lambda query: cluster_biased_retrieve(vectorstore, query, k=k))

def build_rag_chain(vectorstore, chat_model, k=2):
    retriever = make_cluster_retriever(vectorstore, k=k)
    return (
        {"context": retriever | format_docs, "question": lambda x: x}
        | rag_prompt
        | chat_model
        | StrOutputParser()
    )

def build_hint_chain(vectorstore, chat_model, k=2):
    retriever = make_cluster_retriever(vectorstore, k=k)
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

def build_coaching_chain(vectorstore, chat_model, max_cards=15):
    retriever = RunnableLambda(lambda query: get_cluster_topic_docs(vectorstore, query, max_cards=max_cards))
    return (
        {"context": retriever | format_docs, "question": lambda x: x}
        | coaching_prompt
        | chat_model
        | StrOutputParser()
    )