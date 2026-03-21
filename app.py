import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompts import system_prompt

app = Flask(__name__)

load_dotenv()

def build_rag_chain():
    google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not google_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")

    os.environ["GOOGLE_API_KEY"] = google_api_key

    embeddings = download_hugging_face_embeddings()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Change this line in build_rag_chain:
    chat_model = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), # Try removing "-latest"
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


rag_chain = None
startup_error = None

try:
    rag_chain = build_rag_chain()
except Exception as exc:
    startup_error = str(exc)
    app.logger.exception("Failed to initialize the RAG chain.")

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return "Please send a message."

    if rag_chain is None:
        return f"Application startup failed: {startup_error}", 500

    try:
        response = rag_chain.invoke({"input": msg})
        return str(response.get("answer", "Error processing your request."))
    except Exception as e:
        print("Error:", e)
        return "Error processing your request."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
