
from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms.ollama import ChatOllama        # updated import
from langchain.embeddings import HuggingFaceEmbeddings  # updated import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Initialize app
app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY, "Missing Pinecone API Key"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

index_name = "medical-chatbot"
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        api_key=PINECONE_API_KEY,
    )
except Exception as e:
    raise RuntimeError(f"Error initializing Pinecone index: {e}")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOllama(model="qwen2:0.5b", temperature=0.7)
system_prompt = "You are a helpful and knowledgeable medical assistant."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
