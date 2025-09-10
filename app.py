
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

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

# Use this instead for lowest memory usage
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 4,
        'show_progress_bar': False  # Reduces memory overhead
    }
)

# Connect to existing Pinecone index
index_name = "medical-chatbot"
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
except Exception as e:
    raise RuntimeError(f"Error initializing Pinecone index: {e}")

# Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Ollama LLM
chatModel = Ollama(model="qwen2:0.5b", temperature=0.7)
system_prompt = "You are a helpful and knowledgeable medical assistant."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg")
        if not msg:
            return "No message provided", 400
        response = rag_chain.invoke({"input": msg})
        return str(response.get("answer", "No answer"))
    except Exception as e:
        print("Error:", e)
        return f"Error: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
