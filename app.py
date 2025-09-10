
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import time
import gc
import traceback

# LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms.ollama import Ollama

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

# Global variables for lazy loading
_embeddings = None
_docsearch = None
_retriever = None
_chatModel = None
_rag_chain = None
_last_activity_time = time.time()
_initialization_error = None # New variable to store initialization errors

# --- Lazy Loading Functions ---
def get_embeddings():
    """Lazy load HuggingFace embeddings."""
    global _embeddings, _last_activity_time
    if _embeddings is None:
        try:
            print("Initializing HuggingFace Embeddings...")
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 4},
                show_progress=False
            )
            print("Embeddings initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {e}")
    _last_activity_time = time.time()
    return _embeddings


def get_docsearch():
    """Lazy load Pinecone vector store."""
    global _docsearch, _last_activity_time
    if _docsearch is None:
        try:
            print("Connecting to Pinecone index...")
            embeddings = get_embeddings()
            _docsearch = PineconeVectorStore.from_existing_index(
                index_name="medical-chatbot",
                embedding=embeddings
            )
            print("Pinecone connection successful.")
        except Exception as e:
            # Provide more context for Pinecone errors
            raise RuntimeError(f"Failed to connect to Pinecone index 'medical-chatbot': {e}")
    _last_activity_time = time.time()
    return _docsearch


def get_retriever():
    """Lazy load retriever."""
    global _retriever, _last_activity_time
    if _retriever is None:
        print("Creating retriever...")
        docsearch = get_docsearch()
        _retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        print("Retriever created.")
    _last_activity_time = time.time()
    return _retriever


def get_chat_model():
    """Lazy load the Ollama chat model."""
    global _chatModel, _last_activity_time
    if _chatModel is None:
        try:
            print("Connecting to Ollama model 'qwen2:0.5b'...")
            _chatModel = Ollama(model="qwen2:0.5b", temperature=0.7)
            # A simple invoke to test the connection and model existence
            _chatModel.invoke("test message") 
            print("Ollama model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Ollama model 'qwen2:0.5b': {e}")
    _last_activity_time = time.time()
    return _chatModel


def get_rag_chain():
    """Lazy load the RAG chain."""
    global _rag_chain, _last_activity_time
    if _rag_chain is None:
        print("Creating RAG chain...")
        retriever = get_retriever()
        chatModel = get_chat_model()

        # The prompt must contain a {context} placeholder for the retrieved documents.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and knowledgeable medical assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        _rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG chain created successfully.")

    _last_activity_time = time.time()
    return _rag_chain


def cleanup_memory():
    """Clear cached models to free up memory after period of inactivity."""
    global _embeddings, _docsearch, _retriever, _chatModel, _rag_chain
    current_time = time.time()

    # Clean up after 10 minutes of inactivity
    if current_time - _last_activity_time > 600:
        print("Memory clean-up initiated due to inactivity.")
        _embeddings = None
        _docsearch = None
        _retriever = None
        _chatModel = None
        _rag_chain = None
        gc.collect()
        print("Memory cleaned up.")


# --- Flask Routes ---

@app.route("/")
def index():
    cleanup_memory()
    return render_template("chat.html")


@app.route("/health")
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "memory_allocated": _rag_chain is not None
    })

@app.route("/rag_status")
def rag_status():
    """New endpoint to check the status of RAG components."""
    status = {}
    try:
        # Check environment variables
        if not os.getenv("PINECONE_API_KEY"):
            return jsonify({"status": "error", "error": "PINECONE_API_KEY is not set in .env file."}), 500
        
        # Test Pinecone connection
        get_docsearch()
        status["pinecone"] = {"status": "ok", "message": "Connected to Pinecone index."}
    except Exception as e:
        status["pinecone"] = {"status": "error", "message": str(e)}

    try:
        # Test Ollama connection
        get_chat_model()
        status["ollama"] = {"status": "ok", "message": "Ollama model is loaded."}
    except Exception as e:
        status["ollama"] = {"status": "error", "message": str(e)}
        
    return jsonify(status)


@app.route("/get", methods=["POST"])
def chat():
    global _initialization_error
    try:
        data = request.get_json(force=True)
        msg = data.get("msg") if data else None

        if not msg:
            return jsonify({"error": "No message provided"}), 400

        if len(msg) > 500:
            return jsonify({"error": "Message too long. Maximum 500 characters."}), 400

        # Attempt to get the RAG chain, this will trigger lazy loading
        rag_chain = get_rag_chain()
        
        print(f"Invoking RAG chain with message: '{msg}'")
        response = rag_chain.invoke({"input": msg})
        print("RAG chain invocation successful.")

        # LangChain's response object can be complex.
        # This part ensures we get the final string result.
        answer = response.get("answer")
        if not answer:
             raise ValueError("RAG chain response did not contain an 'answer' field.")
             
        return jsonify({"answer": answer})

    except Exception as e:
        print("--- RAG CHAT ERROR ---")
        print(f"An exception occurred: {e}")
        traceback.print_exc()
        print("-----------------------")
        return jsonify({"error": f"An internal server error occurred. Please check the server logs for more details. Error: {e}"}), 500


@app.teardown_request
def teardown_request(exception=None):
    cleanup_memory()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)
