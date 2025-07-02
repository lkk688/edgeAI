# 📚 RAG Applications with LangChain on Jetson

## 🤔 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a method to enhance LLMs with external knowledge. Instead of relying only on pre-trained knowledge, RAG retrieves relevant documents and feeds them into the prompt.

### 💡 Why RAG on Jetson?

* Jetson can run LLMs locally
* Useful for querying private/local data
* Lightweight RAG can power edge AI assistants without internet

---

## 🔧 RAG Architecture

1. **User Query** →
2. **Retriever** (Vector DB) finds relevant chunks →
3. **Combiner/Prompt Template** builds context →
4. **LLM (llama.cpp / Ollama)** generates answer

---

## 🧱 Key Components in LangChain

* `DocumentLoader`: Load PDFs, markdowns, text
* `TextSplitter`: Break documents into chunks
* `Embeddings`: Convert text to vectors
* `VectorStore`: FAISS, Chroma, Qdrant, etc.
* `Retriever`: Pull relevant chunks
* `LLM`: Generate final answer using GGUF models via llama-cpp or Ollama

---

## 🧪 Lab: Build RAG App with Multiple Backends on Jetson

### 🧰 Setup

```bash
pip install langchain llama-cpp-python chromadb faiss-cpu qdrant-client sentence-transformers
```

### 🔹 Step 1: Load and Split Document

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("data/jetson_guide.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

### 🔹 Step 2: Embed and Index (Choose Backend)

#### Option A: ChromaDB

```python
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="db_chroma")
```

#### Option B: FAISS

```python
from langchain.vectorstores import FAISS
faiss_store = FAISS.from_documents(chunks, embedding)
```

#### Option C: Qdrant (self-hosted or remote)

```python
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")
qdrant_store = Qdrant.from_documents(chunks, embedding, client=client, collection_name="jetson_docs")
```

Convert any vector store to retriever:

```python
retriever = vectorstore.as_retriever()
```

---

### 🔹 Step 3: RAG with Multiple Model Inference Backends

#### ✅ llama-cpp Backend (Local GGUF Model)

```python
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

llm = LlamaCpp(model_path="/models/mistral.gguf", n_gpu_layers=80)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print(qa.run("What is Jetson Orin Nano used for?"))
```

#### ✅ Ollama Backend (Local REST API)

```python
from langchain.llms import OpenAI
ollama_llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
qa_ollama = RetrievalQA.from_chain_type(llm=ollama_llm, retriever=retriever)
print(qa_ollama.run("What is Jetson Orin Nano used for?"))
```

---

## 📋 Lab Deliverables

* Run the same query across multiple vector DBs and model backends
* Record differences in:

  * Latency
  * Answer quality
  * Memory usage
* Submit a table comparing results

---

## 💡 Use Cases for Jetson Edge

* Campus FAQ bots with private syllabus
* On-device document search (manuals, code docs)
* Assistive RAG chatbot with no internet

---

## ✅ Summary

* RAG augments LLMs with context-aware search
* Vector DB options: Chroma, FAISS, Qdrant (all lightweight and Jetson-compatible)
* Inference backends: llama.cpp and Ollama, both support GGUF models
* Jetson can handle small-to-medium scale RAG locally with optimized models

→ Next: [Local AI Agents](10_local_ai_agents_jetson.md)
