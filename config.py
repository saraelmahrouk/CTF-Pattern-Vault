import os
from dotenv import load_dotenv
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
CHAT_MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
CHAT_PROVIDER = "together"
EMBEDDING_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"

response_schemas = [
    ResponseSchema(name="vulnerability_class", description="e.g. 'PDF injection' or 'weak cipher mode'"),
    ResponseSchema(name="tools_used", description="list of tool names used"),
    ResponseSchema(name="difficulty", description="one of: beginner, intermediate, advanced"),
    ResponseSchema(name="key_insight", description="one sentence describing the pivotal realization"),
]

hint_response_schemas = [
    ResponseSchema(
        name="category_hint",
        description="A vague, high-level nudge naming only the general vulnerability category or area (e.g. 'this looks like an encoding problem'). Do NOT mention specific tools or techniques."
    ),
    ResponseSchema(
        name="tool_hint",
        description="A more specific nudge mentioning relevant tools or techniques to try, without giving away the full solution path."
    ),
    ResponseSchema(
        name="walkthrough_hint",
        description="A near-complete walkthrough of the solving approach, stopping just short of stating the final flag."
    ),
]

hint_parser = StructuredOutputParser.from_response_schemas(hint_response_schemas)
hint_format_instructions = hint_parser.get_format_instructions()

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()