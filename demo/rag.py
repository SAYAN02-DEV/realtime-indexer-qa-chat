import os
from dotenv import load_dotenv
from llama_index.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.llms.types import ChatMessage, MessageRole
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import PathwayRetriever
from traceloop.sdk import Traceloop
from pathway.xpacks.llm.vector_store import VectorStoreClient

# Gemini imports
import google.generativeai as genai
from llama_index.llms.gemini import Gemini

# Load environment variables
load_dotenv()

# Initialize Traceloop
Traceloop.init(app_name=os.environ.get("APP_NAME", "PW - LlamaIndex (Streamlit)"))

# Pathway host/port
DEFAULT_PATHWAY_HOST = "demo-document-indexing.pathway.stream"
PATHWAY_HOST = os.environ.get("PATHWAY_HOST", DEFAULT_PATHWAY_HOST)
PATHWAY_PORT = int(os.environ.get("PATHWAY_PORT", "80"))

def get_additional_headers():
    key = os.environ.get("PATHWAY_API_KEY")
    return {"X-Pathway-API-Key": key} if key else {}

# Vector store client
vector_client = VectorStoreClient(
    PATHWAY_HOST,
    PATHWAY_PORT,
    additional_headers=get_additional_headers(),
)

# Retriever
retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)
retriever.client = vector_client  # reuse same client

# Configure Gemini dynamically
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
available_models = genai.list_models()
gemini_models = [m["name"] for m in available_models if "gemini" in m["name"].lower()]

if not gemini_models:
    raise ValueError("No Gemini model found. Check your API key or available models.")

selected_model = gemini_models[0]  # pick first available Gemini model
print(f"Using Gemini model: {selected_model}")

llm = Gemini(model=selected_model)

# Retriever query engine
query_engine = RetrieverQueryEngine.from_args(retriever)

# Default chat history
pathway_explanation = "Pathway is a high-throughput, low-latency data processing framework that handles live data & streaming for you."
DEFAULT_MESSAGES = [
    ChatMessage(role=MessageRole.USER, content="What is Pathway?"),
    ChatMessage(role=MessageRole.ASSISTANT, content=pathway_explanation),
]

# Chat engine using Gemini
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=retriever,
    system_prompt="""You are RAG AI that answers users questions based on provided sources.
IF QUESTION IS NOT RELATED TO ANY OF THE CONTEXT DOCUMENTS, SAY IT'S NOT POSSIBLE TO ANSWER USING PHRASE `The looked-up documents do not provide information about...`""",
    verbose=True,
    chat_history=DEFAULT_MESSAGES,
    llm=llm,
)

# Exportable objects
__all__ = ["chat_engine", "query_engine", "vector_client"]
