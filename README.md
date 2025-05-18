

# 🔍 RAG Document Q\&A with Llama 3 + Groq

This is a Streamlit-powered application that allows you to ask questions about two specific research papers — attention.pdf and llm.pdf — using a Retrieval-Augmented Generation (RAG) pipeline powered by the Llama 3 model on Groq's API.

The app loads the documents, splits and embeds the content, creates a vector store with FAISS, and uses LangChain chains to answer natural language queries based on document content.

---

## 🚀 Features

* 📁 Loads only attention.pdf and llm.pdf
* 🔎 Embeds documents using HuggingFace MiniLM
* 🧠 RAG-based QA using Groq’s Llama 3 model (Llama3-8b-8192)
* 🧵 Vector store created using FAISS
* 🖥️ Clean Streamlit UI with real-time Q\&A

---

## 📚 Libraries Used

| Library               | Purpose                                 |
| --------------------- | --------------------------------------- |
| streamlit             | UI and interaction                      |
| langchain             | RAG pipeline and chains                 |
| langchain-groq        | Access to Groq’s Llama 3 models         |
| langchain-huggingface | Text embedding using HuggingFace models |
| langchain-community   | PDF loader and FAISS vector store       |
| python-dotenv         | Load environment variables              |
| openai                | Optional (for compatibility)            |

---

## 📂 Folder Structure

```
project/
├── app.py
├── .env
└── research_papers/
    ├── attention.pdf
    └── llm.pdf
```

---

## ⚙️ Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/your-username/rag-qa-groq.git
cd rag-qa-groq
```

2. Install dependencies

Make sure you’re using Python 3.9+ and install required packages:

```bash
pip install -r requirements.txt
```

If you don’t have a requirements.txt yet, create one with:

```
streamlit
langchain
langchain-groq
langchain-huggingface
langchain-community
python-dotenv
openai
```

3. Create a .env file

```bash
touch .env
```

Paste your API keys into it:

.env

GROQ\_API\_KEY=your\_groq\_api\_key
HF\_TOKEN=your\_huggingface\_token
OPENAI\_API\_KEY=optional\_if\_needed

4. Add your PDFs

Place attention.pdf and llm.pdf inside a folder called research\_papers.

5. Run the app

```bash
streamlit run app.py
```

---

## 🧠 How It Works

* The app loads attention.pdf and llm.pdf using PyPDFLoader.
* The content is chunked using RecursiveCharacterTextSplitter.
* Chunks are embedded using HuggingFace's MiniLM model.
* A FAISS vector store is built from the embeddings.
* A RAG chain is created using Groq's Llama 3 to answer questions.
* User queries retrieve the most relevant passages and generate answers via the LLM.

---

## 🛠️ TODO / Improvements

* Add file upload feature
* Cache vector DB across sessions
* Add support for more documents dynamically
* Improve UI with theming and analytics

---

## 📝 License

MIT License. Free to use and modify.

---

Let me know if you'd like a logo, deployment instructions (Streamlit Cloud, Docker), or a sample output section added!
