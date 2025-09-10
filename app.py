
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import time
import gc

# LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Ollama LLM
from langchain_community.llms.ollama import Ollama

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Pinecone initialization
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY, "Missing Pinecone API Key"

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)  # global initialization

# Global variables for lazy loading
_embeddings = None
_docsearch = None
_retriever = None
_chatModel = None
_rag_chain = None
_last_activity_time = time.time()

def get_embeddings():
    """Lazy load embeddings to save memory"""
    global _embeddings, _last_activity_time
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 4,
                'show_progress_bar': False
            }
        )
    _last_activity_time = time.time()
    return _embeddings

def get_docsearch():
    """Lazy load Pinecone vector store"""
    global _docsearch, _last_activity_time
    if _docsearch is None:
        embeddings = get_embeddings()
        _docsearch = PineconeVectorStore.from_existing_index(
            index_name="medical-chatbot",
            embedding=embeddings
        )
    _last_activity_time = time.time()
    return _docsearch

def get_retriever():
    """Lazy load retriever"""
    global _retriever, _last_activity_time
    if _retriever is None:
        docsearch = get_docsearch()
        _retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})  # Reduced to k=2
    _last_activity_time = time.time()
    return _retriever

def get_chat_model():
    """Lazy load the chat model"""
    global _chatModel, _last_activity_time
    if _chatModel is None:
        _chatModel = Ollama(model="qwen2:0.5b", temperature=0.7)
    _last_activity_time = time.time()
    return _chatModel

def get_rag_chain():
    """Lazy load the RAG chain"""
    global _rag_chain, _last_activity_time
    if _rag_chain is None:
        retriever = get_retriever()
        chatModel = get_chat_model()
        
        system_prompt = "You are a helpful and knowledgeable medical assistant."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        _rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    _last_activity_time = time.time()
    return _rag_chain

def cleanup_memory():
    """Clear cached models to free up memory after period of inactivity"""
    global _embeddings, _docsearch, _retriever, _chatModel, _rag_chain
    current_time = time.time()
    
    # Clean up after 10 minutes of inactivity
    if current_time - _last_activity_time > 600:
        _embeddings = None
        _docsearch = None
        _retriever = None
        _chatModel = None
        _rag_chain = None
        gc.collect()  # Force garbage collection
        print("Memory cleaned up due to inactivity")

# Flask routes
@app.route("/")
def index():
    cleanup_memory()  # Check if we need to clean up before serving
    return render_template("chat.html")

@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "memory_allocated": _embeddings is not None
    })

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg")
        if not msg:
            return jsonify({"error": "No message provided"}), 400
        
        # Limit input length to prevent excessive processing
        if len(msg) > 500:
            return jsonify({"error": "Message too long. Maximum 500 characters."}), 400
        
        # Get the RAG chain (will initialize components if needed)
        rag_chain = get_rag_chain()
        
        # Process the query
        response = rag_chain.invoke({"input": msg})
        
        return jsonify({"answer": str(response.get("answer", "No answer"))})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "An internal error occurred"}), 500

# Add a cleanup function
@app.teardown_request
def teardown_request(exception=None):
    cleanup_memory()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Use a single worker and enable reloader only in development
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)