"""Gradio app entrypoint for Hugging Face Spaces deployment."""

from groq import Groq
import gradio as gr

from answer import ask_groq, build_document_frequency, is_greeting, load_chunks, retrieve
from config import GROQ_API_KEY, TOP_K


CHUNKS = load_chunks()
DF = build_document_frequency(CHUNKS) if CHUNKS else {}
CLIENT = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def respond(message: str, history):
    question = (message or "").strip()
    if not question:
        return "Please enter a question."

    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY. Add it as a Secret in Hugging Face Space settings."

    if not CHUNKS:
        return "No local store found. Ensure rag_store.json is committed or run ingest first."

    if is_greeting(question):
        return "Hey! Ask me anything about Insurellm, contracts, products, or employees."

    top = retrieve(question, CHUNKS, TOP_K, DF)
    if not top:
        return "I could not find relevant context."

    try:
        answer = ask_groq(CLIENT, question, top)
    except Exception as exc:
        return f"Error while contacting Groq: {exc}"

    sources = "\n".join([f"- {item['source']} (score={item['score']:.2f})" for item in top])
    return f"{answer}\n\nSources:\n{sources}"


demo = gr.ChatInterface(
    fn=respond,
    title="RAG Knowledge Worker",
    description="Ask questions about your knowledge base using Groq-powered retrieval and generation.",
)


if __name__ == "__main__":
    demo.launch()
